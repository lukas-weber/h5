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
 * @brief Implementation details for conversion.hpp.
 */

#include "./conversion.hpp"
#include "./object.hpp"

#include <hdf5.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace h5 {

  namespace {

    // Storage for the library-global conversion policy.
    std::atomic<conversion> &policy_ref() {
      static std::atomic<conversion> policy = conversion::exact;
      return policy;
    }

    // Check whether the source element of a native floating point datatype is an infinity.
    template <typename T>
    bool is_infinite_as(::hid_t src_id, ::hid_t native_id, void const *src_buf) {
      if (H5Tequal(src_id, native_id) <= 0) return false;
      T x{};
      std::memcpy(&x, src_buf, sizeof(T));
      return std::isinf(x);
    }

    bool is_infinite(::hid_t src_id, void const *src_buf) {
      return is_infinite_as<float>(src_id, H5T_NATIVE_FLOAT, src_buf) || is_infinite_as<double>(src_id, H5T_NATIVE_DOUBLE, src_buf)
         || is_infinite_as<long double>(src_id, H5T_NATIVE_LDOUBLE, src_buf);
    }

    // Conversion exception callback installed on the global transfer property list: abort whenever a value cannot be
    // represented in the destination datatype.
    //
    // The extra cases are necessary to make sure we can still read infs and nans in all cases. 
    H5T_conv_ret_t abort_on_value_loss(H5T_conv_except_t except_type, ::hid_t src_id, ::hid_t dst_id, void *src_buf, void * /*dst_buf*/,
                                       void * /*user_data*/) {
      if (H5Tget_class(dst_id) == H5T_FLOAT) {
        bool const is_special = (except_type == H5T_CONV_EXCEPT_PINF || except_type == H5T_CONV_EXCEPT_NINF || except_type == H5T_CONV_EXCEPT_NAN);
        bool const is_range   = (except_type == H5T_CONV_EXCEPT_RANGE_HI || except_type == H5T_CONV_EXCEPT_RANGE_LOW);
        if (is_special || (is_range && is_infinite(src_id, src_buf))) return H5T_CONV_UNHANDLED;
      }
      return H5T_CONV_ABORT;
    }

    // Create the global dataset transfer property list and install the conversion callback.
    proplist make_transfer_plist() {
      proplist p = H5Pcreate(H5P_DATASET_XFER);
      if (!p.is_valid()) throw std::runtime_error("Error in h5: Creating the global dataset transfer property list failed");
      if (H5Pset_type_conv_cb(p, &abort_on_value_loss, nullptr) < 0)
        throw std::runtime_error("Error in h5: H5Pset_type_conv_cb on the global dataset transfer property list failed");
      return p;
    }

  } // namespace

  void set_conversion_policy(conversion c) { policy_ref() = c; }

  conversion get_conversion_policy() { return policy_ref(); }

  namespace detail {

    proplist &transfer_plist() {
      static proplist p = make_transfer_plist();
      return p;
    }

    void check_conversion(datatype src, datatype dst, std::string const &context) {
      if (get_conversion_policy() == conversion::exact && !hdf5_type_equal(src, dst)) {
        throw std::runtime_error(context + ": HDF5 type mismatch " + get_name_of_h5_type(src) + " -> " + get_name_of_h5_type(dst)
                                 + " (conversion policy is h5::conversion::exact)");
      }
    }

  } // namespace detail

} // namespace h5
