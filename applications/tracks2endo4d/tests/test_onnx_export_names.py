# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import unittest


class OnnxExportNameTests(unittest.TestCase):
    def test_export_input_and_output_names_do_not_collide(self):
        exporter_path = (
            pathlib.Path(__file__).resolve().parents[1] / "onnx_conversion" / "onnx_tapnext.py"
        )
        module = ast.parse(exporter_path.read_text(encoding="utf-8"))

        checked_exports = 0
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "export"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "onnx"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "torch"
            ):
                continue

            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            input_names = ast.literal_eval(keywords["input_names"])
            output_names = ast.literal_eval(keywords["output_names"])
            collisions = set(input_names) & set(output_names)

            self.assertFalse(
                collisions,
                f"{exporter_path} exports ONNX bindings as both inputs and outputs: "
                f"{sorted(collisions)}",
            )
            checked_exports += 1

        self.assertEqual(checked_exports, 2)


if __name__ == "__main__":
    unittest.main()
