#pragma once

// =====================================================================================
// LES SOLVEURS LINEAIRES : `L d = b` sur le systeme reduit, la jauge `d_0 = 0` etant a leur
// charge. Tous derivent de `Lineaire` et n'offrent que ca :
//
//     bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d );
//     const char *nom() const;
//     StatsLin st;        // le temps par poste, et ce que le solveur a fait
//
// L'INTERFACE EST VIRTUELLE, ET C'EST UN CHOIX DE COMPILATION. Elle etait un parametre de
// template, ce qui multipliait par trois -- un par solveur -- tout ce qui vit sous `Newton` :
// `main_image.cpp` s'instanciait quarante-huit fois ( 2 noyaux x 4 capacites x 2 mesures x
// 3 solveurs ) et mettait huit minutes et quatre gigaoctets a compiler. Le prix d'un appel
// virtuel se paie UNE FOIS PAR RESOLUTION d'un systeme a `n` inconnues : il est sous le bruit,
// et il n'y a rien a inliner dans un appel qui dure des millisecondes. Les REGLAGES, eux,
// restent sur le type concret -- ils se posent dans `main` avant que le template ne commence.
//
// Le solveur lineaire n'est pas l'objet de l'etude et il ne faut pas qu'il le devienne : on prend
// des bibliotheques en-tetes seuls.
//
//   AMG       AMGCL, multigrille algebrique + CG. Le bon outil pour ce systeme : la hessienne est
//             le laplacien d'un graphe presque planaire, le cout reste en O( n ) et la hierarchie
//             se construit en parallele ( OpenMP ). 4.9x sur le total a n=1e6 contre Cholesky.
//             Trois hierarchies, parce que le nuage decide : l'agregation lissee suppose des poids
//             d'aretes comparables, ce que le nuage de lignes viole ; Ruge-Stuben, qui choisit ses
//             noeuds grossiers arete par arete, y divise les iterations par trois.
//   CHOLESKY  Eigen `SimplicialLDLT`, renumerotation AMD, l'analyse symbolique refaite SEULEMENT
//             quand le motif change. Scalaire et sequentiel, mais il gagne encore sur le nuage de
//             lignes a n=1e5 ; c'est le temoin.
// =====================================================================================

#include "solver/Laplacien.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>

#if __has_include( <amgcl/make_solver.hpp> )
#  define SF_AMGCL 1
#  include <amgcl/adapter/crs_tuple.hpp>
#  include <amgcl/amg.hpp>
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/coarsening/ruge_stuben.hpp>
#  include <amgcl/coarsening/smoothed_aggregation.hpp>
#  include <amgcl/make_solver.hpp>
#  include <amgcl/relaxation/gauss_seidel.hpp>
#  include <amgcl/relaxation/spai0.hpp>
#  include <amgcl/solver/cg.hpp>
#endif

#if __has_include( <Eigen/SparseCholesky> )
#  define SF_EIGEN 1
#  include <Eigen/SparseCholesky>
#  include <Eigen/SparseCore>
#endif

namespace sf {

/// ce qu'un solveur lineaire rapporte, cumule sur toutes ses resolutions.
struct StatsLin {
    double t_forme = 0;        ///< la mise en forme de la matrice ( CRS, triplets )
    double t_hier  = 0;        ///< la hierarchie AMG, ou l'analyse symbolique
    double t_res   = 0;        ///< la resolution proprement dite ( ou factorisation + descente )
    int    nb_hier = 0;        ///< combien de hierarchies / analyses
    int    nb_iter = 0;        ///< iterations de CG, en tout
    TF     pire    = 0;        ///< le pire residu relatif rendu
    double total() const { return t_forme + t_hier + t_res; }
};

/// CE QU'UN SOLVEUR LINEAIRE DOIT SAVOIR FAIRE, et rien de plus.
struct Lineaire {
    StatsLin st;                   ///< cumule sur toutes les resolutions

    virtual ~Lineaire() = default;
    virtual const char *nom() const = 0;
    virtual bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) = 0;

    /// UNE RESOLUTION DE PLUS sur la derniere hierarchie / factorisation, pour qui reutilise un
    /// laplacien fige ( `PremierOrdre.h` ). `sait_encore()` dit si ca a un sens maintenant.
    virtual bool sait_encore() const { return false; }
    virtual void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) { (void) b; (void) d; }

    /// l'ordre de l'arbre ( `pd.ids` ), que seul le multigrille maison utilise : son agregation
    /// est la tranche de rangs, donc elle est GRATUITE -- mais encore faut-il lui donner les rangs.
    virtual void ordre( const std::int32_t *ids, SI nb ) { (void) ids; (void) nb; }
};

#ifdef SF_AMGCL
struct Amg : Lineaire {
    enum Variante : int { SA_SPAI0 = 0, SA_GS = 1, RS_GS = 2 };
    int      variante = SA_SPAI0;
    TF       tol      = 1e-10;     ///< residu RELATIF
    int      maxit    = 20000;
    // LA HIERARCHIE PEUT SERVIR PLUSIEURS FOIS. Entre deux iterations de Newton la hessienne
    // change, mais son graphe bouge a peine -- quelques aretes. Or un PRECONDITIONNEUR n'a pas
    // besoin d'etre exact : seule la matrice que voit le CG doit l'etre, et on la reconstruit a
    // chaque fois. `make_solver` d'amgcl expose exactement cette dissociation par sa surcharge
    // `( A, rhs, x )` -- resoudre avec une matrice NEUVE contre la hierarchie deja montee.
    // MAIS CA NE MARCHE PAS ICI, ET LA MESURE EST NETTE : a `n = 1e5`, `refaire 4` fait tomber la
    // montee de 13.4 a 3.4 s et monte la resolution de 21.9 a 41.1 -- les iterations de CG passent
    // de 9369 a 16764, soit +79 %. Total 93.9 s contre 103.0.
    //
    // La raison se lit dans notre propre implementation : `Mg::rebranche` RECALCULE les
    // coefficients du lisseur au niveau FIN avec les nouvelles valeurs, et n'y perd que 11 %
    // d'iterations. Amgcl garde les siens -- son `spai0` du niveau fin reste celui de l'ancienne
    // matrice -- et il n'y a pas d'API pour le rafraichir seul. La dissociation entre la matrice
    // du CG et celle du preconditionneur ne paie donc que si on peut rafraichir le niveau fin,
    // c'est-a-dire si on possede le solveur. Defaut : 1, et l'option reste pour qui veut verifier.
    int      refaire  = 1;         ///< la hierarchie refaite toutes les `refaire` resolutions
    /// la derniere hierarchie, gardee pour `resout_encore` ( le type du solveur depend de la variante )
    std::function<std::tuple<int,double>( const std::vector<double> &, std::vector<double> & )> encore;
    /// ... et la meme, mais avec UNE AUTRE MATRICE : c'est elle qui sert a `refaire`
    std::function<std::tuple<int,double>( std::vector<int> &, std::vector<int> &, std::vector<double> &,
                                          const std::vector<double> &, std::vector<double> & )> avec_A;
    int      depuis   = 0;         ///< resolutions depuis la derniere montee
    SI       n_prec   = 0;         ///< la taille de la derniere hierarchie

    const char *nom() const override {
        return variante == RS_GS ? "AMGCL Ruge-Stuben+GS"
             : variante == SA_GS ? "AMGCL agregation+GS" : "AMGCL agregation+spai0";
    }

    /// UNE RESOLUTION DE PLUS sur la derniere hierarchie : la matrice a change, pas ses voisinages,
    /// et c'est la hierarchie qui coute -- pour qui reutilise un laplacien fige ( `PremierOrdre.h` ).
    bool sait_encore() const override { return bool( encore ); }
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) override {
        const SI n = SI( b.size() ), m = n - 1;
        const double t0 = now();
        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i ) rb[ i - 1 ] = double( b[ i ] );
        auto [ it, err ] = encore( rb, sol );
        st.nb_iter += it;
        st.pire = std::max( st.pire, TF( err ) );
        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i ) d[ i ] = TF( sol[ i - 1 ] );
        st.t_res += now() - t0;
    }

    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) override {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<int> ptr, col;
        std::vector<double> val;
        L.crs_reduit( ptr, col, val );
        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        const double t1 = now();
        st.t_forme += t1 - t0;

        using Back = amgcl::backend::builtin<double>;
        int it = 0;
        double err = 0;
        auto lance = [ & ]( auto tag ) {
            using Solv = typename decltype( tag )::type;
            typename Solv::params prm;
            prm.solver.tol = double( tol );
            prm.solver.maxiter = maxit;
            auto so = std::make_shared<Solv>( std::tie( m, ptr, col, val ), prm );
            const double ta = now();
            st.t_hier += ta - t1;
            ++st.nb_hier;
            std::tie( it, err ) = ( *so )( rb, sol );
            st.t_res += now() - ta;
            encore = [ so ]( const std::vector<double> &b, std::vector<double> &x ) { return ( *so )( b, x ); };
            // LA MEME HIERARCHIE, UNE MATRICE NEUVE. On passe par `Back::matrix` plutot que par
            // l'adaptateur de tuple : l'adaptateur sait CONSTRUIRE une hierarchie, le CG a besoin
            // d'un operateur qui sait faire un produit matrice-vecteur.
            avec_A = [ so ]( std::vector<int> &p, std::vector<int> &c, std::vector<double> &v,
                             const std::vector<double> &bb, std::vector<double> &x ) {
                const SI mm = SI( p.size() ) - 1;
                typename Back::matrix Am( std::tie( mm, p, c, v ) );
                int i2 = 0; double e2 = 0;
                std::tie( i2, e2 ) = ( *so )( Am, bb, x );
                return std::make_tuple( i2, e2 );
            };
        };
        using SaSpai = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>, amgcl::solver::cg<Back>>;
        using SaGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        using RsGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::ruge_stuben, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        if ( avec_A && n == n_prec && depuis < std::max( refaire, 1 ) ) {
            const double ta = now();
            std::tie( it, err ) = avec_A( ptr, col, val, rb, sol );
            st.t_res += now() - ta;
            ++depuis;
        } else {
            if      ( variante == SA_GS ) lance( std::type_identity<SaGs>{} );
            else if ( variante == RS_GS ) lance( std::type_identity<RsGs>{} );
            else                          lance( std::type_identity<SaSpai>{} );
            depuis = 1;
            n_prec = n;
        }
        st.nb_iter += it;
        st.pire = std::max( st.pire, TF( err ) );

        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = TF( sol[ i - 1 ] );
        return err < 1;                                  // `1` : le solveur n'a rien fait du tout
    }
};
#endif

#ifdef SF_EIGEN
struct Cholesky : Lineaire {
    using SpM = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    std::vector<SI> motif;         ///< le motif de la derniere analyse symbolique

    const char *nom() const override { return "Cholesky creux ( Eigen LDLT, AMD )"; }

    /// UNE DESCENTE DE PLUS sur la derniere factorisation : pour qui a plusieurs seconds membres.
    bool sait_encore() const override { return ! motif.empty(); }
    void resout_encore( const std::vector<TF> &b, std::vector<TF> &d ) override {
        const SI n = SI( b.size() ), m = n - 1;
        const double t0 = now();
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i ) rb[ i - 1 ] = double( b[ i ] );
        const Eigen::VectorXd sol = so.solve( rb );
        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i ) d[ i ] = TF( sol[ i - 1 ] );
        st.t_res += now() - t0;
    }

    /// LE MOTIF NE BOUGE PRESQUE PAS : quelques aretes par iteration au debut, zero a la fin, alors
    /// que la renumerotation et l'analyse symbolique coutent le tiers de la factorisation. On les
    /// refait SEULEMENT quand le motif a change -- compare tel quel, une passe lineaire.
    bool resout( const Laplacien &L, const std::vector<TF> &b, std::vector<TF> &d ) override {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve( size_t( L.row[ n ] ) / 2 + n );
        for ( SI i = 1; i < n; ++i ) {
            tri.emplace_back( i - 1, i - 1, double( L.dia[ i ] ) );
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( j >= 1 && j < i )                   // le triangle INFERIEUR seul
                    tri.emplace_back( i - 1, j - 1, -double( L.c[ k ] ) );
            }
        }
        SpM A( m, m );
        A.setFromTriplets( tri.begin(), tri.end() );
        const double t1 = now();
        st.t_forme += t1 - t0;

        std::vector<SI> mot( A.outerIndexPtr(), A.outerIndexPtr() + m + 1 );
        mot.insert( mot.end(), A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros() );
        if ( mot != motif ) {
            so.analyzePattern( A );
            motif.swap( mot );
            ++st.nb_hier;
        }
        const double t2 = now();
        st.t_hier += t2 - t1;

        so.factorize( A );
        if ( so.info() != Eigen::Success ) { st.t_res += now() - t2; return false; }
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = double( b[ i ] );
        const Eigen::VectorXd sol = so.solve( rb );
        st.t_res += now() - t2;
        if ( so.info() != Eigen::Success )
            return false;

        d.assign( n, TF( 0 ) );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = TF( sol[ i - 1 ] );
        return true;
    }
};
#endif

} // namespace sf
