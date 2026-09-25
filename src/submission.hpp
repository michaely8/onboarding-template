#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>
#include <omp.h>
#include <sched.h>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0) {};

  std::size_t rows() const {
    return rows_;
  };
  std::size_t cols() const {
    return cols_;
  };

  double* data() {
    return data_.data();
  };
  const double* data() const {
    return data_.data();
  };

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i * cols_ + j];
  };
  double  operator()(std::size_t i, std::size_t j) const {
    return data_[i * cols_ + j];
  };
};

inline void pin_threads() {
  static bool pinned = false;
  if (pinned) {
    return;
  }

  pinned = true;

#pragma omp parallel
  {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(omp_get_thread_num(), &set);
    sched_setaffinity(0, sizeof(set), &set);
  }
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  const double* old_data = old_grid.data();
  double* new_data = new_grid.data();

  if (rows == 0 || cols == 0) {
    return;
  }

  pin_threads();

#pragma omp parallel for schedule(static)
  for (std::size_t i = cols; i < cols * (rows - 1); i += cols) {
    for (std::size_t j = 1; j < cols - 1; j++) {
      const std::size_t val = i + j;
      new_data[val] = 0.5 * old_data[val] + 0.125 * (old_data[val - cols] + old_data[val + cols] + old_data[val - 1] + old_data[val + 1]);
    }

    new_data[i] = old_data[i];
    new_data[i + cols - 1] = old_data[i + cols - 1];
  }

  std::copy(old_data, old_data + cols, new_data);
  std::copy(old_data + cols * (rows - 1), old_data + cols * rows, new_data + cols * (rows - 1));
};
