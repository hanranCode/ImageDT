# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standalone LeNet compatibility tests; the main suite lives under tests/."""

import unittest

from imagedt._backends import require

from .lenet import lenet


class LeNetTest(unittest.TestCase):
    def testBuild(self):
        tf = require("tensorflow", "tensorflow")
        logits, endpoints = lenet(tf.zeros((2, 28, 28, 3)), 10, is_training=False)
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertEqual(
            set(endpoints),
            {
                "conv1",
                "pool1",
                "conv2",
                "pool2",
                "Flatten",
                "fc3",
                "dropout3",
                "Logits",
                "Predictions",
            },
        )

    def testForward(self):
        tf = require("tensorflow", "tensorflow")
        _, endpoints = lenet(tf.ones((1, 28, 28, 3)), is_training=False)
        self.assertAlmostEqual(
            float(tf.reduce_sum(endpoints["Predictions"])), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
