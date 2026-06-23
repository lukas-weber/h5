// Copyright (c) 2019-2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Lukas Weber

/**
 * @file
 * @brief Provides the library-global datatype conversion policy.
 */

#ifndef LIBH5_CONVERSION_HPP
#define LIBH5_CONVERSION_HPP

#include "./object.hpp"

#include <string>

namespace h5 {

  /**
   * @addtogroup data_model
   * @{
   */

  /**
   * @brief Datatype conversion policies applied when the HDF5 datatype in the file differs from the one in memory.
   *
   * @details The policy applies in the same way to datasets and attributes, for reading and writing. Note that
   * writing only converts when writing into an existing dataset (h5::array_interface::write_slice), since newly
   * created datasets and attributes always get the in-memory datatype.
   */
  enum class conversion {
    /// Throw a std::runtime_error unless the file and memory datatypes are exactly equal (default).
    exact,
    /// Convert between datatypes, but throw a std::runtime_error if a value cannot be represented in the destination
    /// datatype, i.e. if HDF5 reports that it is out of range, that a fractional part would be truncated, that an
    /// integer loses precision in an integer to floating point conversion or that an infinity/NaN is converted to an
    /// integer.
    checked
  };

  /**
   * @brief Set the library-global datatype conversion policy.
   *
   * @details The policy is a single process-global setting. The default is h5::conversion::exact.
   *
   * @param c Conversion policy.
   */
  void set_conversion_policy(conversion c);

  /**
   * @brief Get the library-global datatype conversion policy.
   * @return Current h5::conversion policy.
   */
  [[nodiscard]] conversion get_conversion_policy();

  namespace detail {

    [[nodiscard]] proplist &transfer_plist();

    void check_conversion(datatype src, datatype dst, std::string const &context);

  } // namespace detail

  /** @} */

} // namespace h5

#endif // LIBH5_CONVERSION_HPP
