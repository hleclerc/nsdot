#pragma once

#include "../support/S.h"
#include "../support/N.h"
#include <algorithm>
#include <cstdint>
#include <string>

namespace asimd {


/**
*/
template<class... Features>
class FeatureSet {
    // extraction of SimdSize from a feature
    template<class Feature,class T,class dummy=N<0>> struct SimdSizeFeature { static constexpr int value = 1; };
    template<class Feature,class T> struct SimdSizeFeature<Feature,T,N<Feature::template SimdSize<T>::value*0>> { static constexpr int value = Feature::template SimdSize<T>::value; };

    // extraction of NbSimdRegisters from a feature
    template<class Feature,class T,int simd_size,class dummy=N<0>> struct NbSimdRegistersFeature { static constexpr int value = 0; };
    template<class Feature,class T,int simd_size> struct NbSimdRegistersFeature<Feature,T,simd_size,N<Feature::template NbSimdRegisters<T,simd_size>::value*0>> { static constexpr int value = Feature::template NbSimdRegisters<T,simd_size>::value; };

    // content if no feature left
    template<class... _Features> struct Content {
        template<class Feature> struct Has { static constexpr bool value = 0; };
        template<class T> struct SimdSize { static constexpr int value = 1; };
        template<class T,int simd_size> struct NbSimdRegisters { static constexpr int value = 0; };
        static std::string feature_names( std::string = "," ) { return ""; }
    };

    // content with at least on feature
    template<class Head,class... Tail> struct Content<Head,Tail...> {
        using Next = Content<Tail...>;

        template<class Feature,int dummy=0> struct Has { enum { value = Next::template Has<Feature>::value }; };
        template<int dummy> struct Has<Head,dummy> { static constexpr bool value = 1; };

        template<class T> struct SimdSize { static constexpr int value = std::max( SimdSizeFeature<Head,T>::value, Next::template SimdSize<T>::value ); };
        template<class T,int simd_size> struct NbSimdRegisters { static constexpr int value = std::max( NbSimdRegistersFeature<Head,T,simd_size>::value, Next::template NbSimdRegisters<T,simd_size>::value ); };

        static std::string feature_names( std::string prefix = "," ) { return prefix + Head::name() + Next::feature_names(); }

        template<class T> auto &value_( S<T> s ) { return next.value_( s ); }
        auto &value_( S<Head> ) { return value; }

        Head value;
        Next next;
    };

    using C = Content<Features...>;
    C content;

public:    
    template<class F>
    struct Has {
        static constexpr bool value = C::template Has<F>::value;
    };

    template<class T>
    struct SimdSize {
        static constexpr int value = C::template SimdSize<T>::value;
    };

    template<class T,int simd_size=SimdSize<T>::value>
    struct SimdAlig {
        static constexpr int value = simd_size * sizeof( T ) * 8; /// in bits
    };

    template<class T,int simd_size=SimdSize<T>::value>
    struct NbSimdRegisters {
        static constexpr int value = C::template NbSimdRegisters<T,simd_size>::value;
    };

    static std::string feature_names() {
        return C::feature_names();
    }

    template<class T>
    auto &value() {
        return content.value_( S<T>() );
    }
};


} //  namespace asimd
