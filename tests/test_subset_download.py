import io
import unittest
import zipfile

from balltrack.subset_download import unpack_member


class MemberTests(unittest.TestCase):
    def test_stored_and_deflated_crc(self):
        for method in [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED]:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w', compression=method) as archive:
                archive.writestr('train/SNMOT-060/img1/000001.jpg', b'image bytes' * 100)
            with zipfile.ZipFile(stream) as archive:
                member = archive.infolist()[0]
                blob = stream.getvalue()[:archive.start_dir]
                self.assertEqual(unpack_member(blob, member), b'image bytes' * 100)
                member.CRC ^= 1
                with self.assertRaises(ValueError):
                    unpack_member(blob, member)

    def test_wrong_member_or_truncation_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('a', b'123456')
        with zipfile.ZipFile(stream) as archive:
            member = archive.infolist()[0]
            blob = stream.getvalue()[:archive.start_dir]
            with self.assertRaises(ValueError):
                unpack_member(blob[:-1], member)
            member.filename = '../wrong'
            with self.assertRaises(ValueError):
                unpack_member(blob, member)
