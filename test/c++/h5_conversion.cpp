// Copyright (c) 2024 Simons Foundation
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

#include <gtest/gtest.h>
#include <h5/h5.hpp>

#include <hdf5.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Write a 1D vector into the slice [0, v.size()) of an existing dataset via the array interface.
template <typename T>
void write_slice(h5::group g, std::string const &name, std::vector<T> const &v) {
  auto const n = static_cast<h5::hsize_t>(v.size());
  h5::array_interface::array_view av(h5::hdf5_type<T>(), (void *)v.data(), 1, false);
  av.slab.count[0]   = n;
  av.parent_shape[0] = n;
  h5::array_interface::hyperslab sl(1, false);
  sl.count[0] = n;
  h5::array_interface::write_slice(g, name, av, sl);
}

// Return the message of the std::runtime_error thrown by f (fails the test if nothing is thrown).
template <typename F>
std::string error_message(F &&f) {
  try {
    f();
  } catch (std::runtime_error const &e) { return e.what(); }
  ADD_FAILURE() << "expected a std::runtime_error";
  return {};
}

// must stay the first test in this file: it checks the default before any other test changes the policy
TEST(H5Conversion, DefaultIsExact) { EXPECT_EQ(h5::get_conversion_policy(), h5::conversion::exact); }

TEST(H5Conversion, ExactAcceptsEqualTypes) {
  h5::set_conversion_policy(h5::conversion::exact);
  h5::file f("conversion.h5", 'w');

  std::vector<int> data{1, 2, 3};
  h5::h5_write(f, "ints", data);
  h5::h5_write_attribute(f, "attr", 7);

  std::vector<int> out;
  int attr = 0;
  EXPECT_NO_THROW(h5::h5_read(f, "ints", out));
  EXPECT_NO_THROW(h5::h5_read_attribute(f, "attr", attr));
  EXPECT_NO_THROW(write_slice(f, "ints", std::vector<int>{4, 5, 6}));
  EXPECT_EQ(out, data);
  EXPECT_EQ(attr, 7);
}

TEST(H5Conversion, ExactRejectsMismatch) {
  h5::set_conversion_policy(h5::conversion::exact);
  h5::file f("conversion.h5", 'w');

  h5::h5_write(f, "ints", std::vector<int>{1, 2, 3});
  h5::h5_write_attribute(f, "attr", 7);

  // even lossless widening conversions are rejected
  std::vector<long> out;
  long attr = 0;
  EXPECT_THROW(h5::h5_read(f, "ints", out), std::runtime_error);
  EXPECT_THROW(h5::h5_read_attribute(f, "attr", attr), std::runtime_error);
  EXPECT_THROW(write_slice(f, "ints", std::vector<long>{4, 5, 6}), std::runtime_error);

  auto const msg = error_message([&] { h5::h5_read(f, "ints", out); });
  EXPECT_NE(msg.find("int"), std::string::npos);
  EXPECT_NE(msg.find("long"), std::string::npos);
}

TEST(H5Conversion, CheckedAllowsWidening) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  h5::h5_write(f, "ints", std::vector<int>{1, 2, 3});
  h5::h5_write_attribute(f, "attr", 7);

  std::vector<long> out;
  long attr = 0;
  EXPECT_NO_THROW(h5::h5_read(f, "ints", out));
  EXPECT_NO_THROW(h5::h5_read_attribute(f, "attr", attr));
  EXPECT_EQ(out, (std::vector<long>{1, 2, 3}));
  EXPECT_EQ(attr, 7);

  std::vector<int> back;
  EXPECT_NO_THROW(write_slice(f, "ints", std::vector<long>{4, 5, 6}));
  h5::h5_read(f, "ints", back);
  EXPECT_EQ(back, (std::vector<int>{4, 5, 6}));
}

TEST(H5Conversion, CheckedAllowsNarrowingIfValuesFit) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  h5::h5_write(f, "ints", std::vector<int>{-3, 0, 3});
  h5::h5_write(f, "doubles", std::vector<double>{0.0, 1.0, -2.0});
  h5::h5_write_attribute(f, "attr", 2.0);

  std::vector<short> shorts;
  std::vector<int> ints;
  int attr = 0;
  EXPECT_NO_THROW(h5::h5_read(f, "ints", shorts));
  EXPECT_NO_THROW(h5::h5_read(f, "doubles", ints));
  EXPECT_NO_THROW(h5::h5_read_attribute(f, "attr", attr));
  EXPECT_EQ(shorts, (std::vector<short>{-3, 0, 3}));
  EXPECT_EQ(ints, (std::vector<int>{0, 1, -2}));
  EXPECT_EQ(attr, 2);
}

TEST(H5Conversion, CheckedRejectsOverflow) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  h5::h5_write(f, "ints", std::vector<int>{1, 100000, 3});
  h5::h5_write_attribute(f, "attr", 100000);

  std::vector<short> out;
  short attr = 0;
  EXPECT_THROW(h5::h5_read(f, "ints", out), std::runtime_error);
  EXPECT_THROW(h5::h5_read_attribute(f, "attr", attr), std::runtime_error);
  EXPECT_THROW(write_slice(f, "ints", std::vector<long>{1, 1L << 40, 3}), std::runtime_error);

  // finite floating point values beyond the destination range are rejected too (unlike infinities)
  h5::h5_write(f, "huge", std::vector<double>{1.0, 1e300});
  std::vector<float> floats;
  EXPECT_THROW(h5::h5_read(f, "huge", floats), std::runtime_error);
}

TEST(H5Conversion, CheckedRejectsTruncation) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  h5::h5_write(f, "fracs", std::vector<double>{0.0, 0.5, 1.0});
  h5::h5_write_attribute(f, "attr", 0.5);

  std::vector<int> out;
  int attr = 0;
  EXPECT_THROW(h5::h5_read(f, "fracs", out), std::runtime_error);
  EXPECT_THROW(h5::h5_read_attribute(f, "attr", attr), std::runtime_error);
}

TEST(H5Conversion, CheckedRejectsIntToFloatPrecisionLoss) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  // 2^24 + 1 is not representable as a float
  h5::h5_write(f, "ints", std::vector<int>{1, (1 << 24) + 1});

  std::vector<float> out;
  EXPECT_THROW(h5::h5_read(f, "ints", out), std::runtime_error);
}

TEST(H5Conversion, CheckedInfAndNan) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  auto const inf = std::numeric_limits<double>::infinity();
  h5::h5_write(f, "special", std::vector<double>{inf, -inf, std::numeric_limits<double>::quiet_NaN()});
  h5::h5_write(f, "inf", std::vector<double>{inf});

  // infinities and NaNs are representable in a float
  std::vector<float> floats;
  EXPECT_NO_THROW(h5::h5_read(f, "special", floats));
  ASSERT_EQ(floats.size(), 3u);
  EXPECT_TRUE(std::isinf(floats[0]) && floats[0] > 0);
  EXPECT_TRUE(std::isinf(floats[1]) && floats[1] < 0);
  EXPECT_TRUE(std::isnan(floats[2]));

  // but not in an int
  std::vector<int> ints;
  EXPECT_THROW(h5::h5_read(f, "inf", ints), std::runtime_error);
}

TEST(H5Conversion, UnconvertibleTypesThrow) {
  h5::set_conversion_policy(h5::conversion::checked);
  h5::file f("conversion.h5", 'w');

  std::vector<int> data{1, 2, 3};
  h5::h5_write(f, "ints", data);

  // there is no conversion path from integer to string, so H5Dread fails and we throw
  h5::datatype str = H5Tcopy(H5T_C_S1);
  H5Tset_size(str, 8);
  std::vector<char> buf(data.size() * 8, '\0');
  h5::array_interface::array_view rv(str, (void *)buf.data(), 1, false);
  rv.slab.count[0]   = static_cast<h5::hsize_t>(data.size());
  rv.parent_shape[0] = static_cast<h5::hsize_t>(data.size());

  EXPECT_THROW(h5::array_interface::read(f, "ints", rv), std::runtime_error);
}
