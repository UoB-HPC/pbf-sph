
#include "omp/ompsph.hpp"
#include "sycl/syclsph.hpp"
#include "sycl/syclsph_2020.hpp"

#ifdef USE_SYCL
template class sph::sycl_impl::Solver<std::size_t, float>;
template class sph::sycl_impl::Solver<std::size_t, double>;
template class sph::sycl2020_impl::Solver<std::size_t, float>;
template class sph::sycl2020_impl::Solver<std::size_t, double>;
#endif

template class sph::omp_impl::Solver<std::size_t, float>;
template class sph::omp_impl::Solver<std::size_t, double>;
