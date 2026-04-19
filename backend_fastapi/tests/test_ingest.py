import tempfile
import unittest
import zipfile
from pathlib import Path

from app.domain.ingest import chunk_text, extract_text


class ChunkingTests(unittest.TestCase):
    def test_no_tiny_tail_chunks_for_regular_text(self) -> None:
        text = ("This is a paragraph with enough content to be chunked safely. " * 200).strip()
        chunks = chunk_text(text, chunk_size=120, overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.strip()) > 40 for chunk in chunks))

    def test_preserves_word_boundaries_in_sliding_chunks(self) -> None:
        text = " ".join(f"word{i}" for i in range(1, 400))
        chunks = chunk_text(text, chunk_size=80, overlap=15)

        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertFalse(chunk.startswith(" "))
            self.assertFalse(chunk.endswith(" "))
            self.assertNotIn("\n\n\n", chunk)

    def test_docx_table_cells_keep_readable_separators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.docx"
            xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Параметр</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Значение</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>FCFS</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>10 мс</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", xml)

            text = extract_text(path)

        self.assertIn("Параметр | Значение", text)
        self.assertIn("FCFS | 10 мс", text)


if __name__ == "__main__":
    unittest.main()
