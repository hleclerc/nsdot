#pragma once

#include "../Queue.h"
#include "../Queue.h"

namespace sdot {

struct CellTag {};

template<class VP>
struct Cell : CellTag {
    using scalar_type = double;
    VP vertex_pos;
};

template<class Batch,class VP>
struct BatchOfCells {
    Batch batch_sizes;
    VP    vertex_pos;
};

// template<class Batch,class VP>
// concept is_a_BatchOfCells = ;

template<typename T>
requires IsCell<T>
void init_as_aligned_hypercube( Queue q, T const &cell, const auto &batch_of_min_pos, const auto &batch_of_max_pos ) {
    cell.vertex_p;
    // q.parallel_for( batch_axes( batch_of_cells, batch_of_min_pos, batch_of_max_pos ), []( auto cell, auto min_pos, auto max_pos ) {

    // } );
}

void init_as_aligdned_hypercube( Queue q, IsCell auto &cell, const auto &batch_of_min_pos, const auto &batch_of_max_pos ) {
    // q.parallel_for( batch_axes( batch_of_cells, batch_of_min_pos, batch_of_max_pos ), []( auto cell, auto min_pos, auto max_pos ) {

    // } );
}

} // namespace sdot
