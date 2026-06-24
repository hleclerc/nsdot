#pragma once

namespace sdot {

struct UndefList { void display( auto &ds ) const { ds << "UndefList"; } };
struct OutList { void display( auto &ds ) const { ds << "OutList"; } };
struct InpList { void display( auto &ds ) const { ds << "InpList"; } };
struct MutList { void display( auto &ds ) const { ds << "MutList"; } };

} // namespace sdot
