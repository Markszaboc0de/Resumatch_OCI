import os
import unittest
from app import app, clean_text
from io import BytesIO

class TestResumeMatcher(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_clean_text(self):
        text = "Hello! This is a TEST with <html> tags."
        cleaned = clean_text(text)
        expected = "hello this is a test with tags"
        self.assertEqual(cleaned, expected)

    # test_cosine_similarity removed as the function was integrated into routes via sklearn

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Welcome to Resumatch", response.data)

    def test_employer_route_get(self):
        response = self.app.get('/employer')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Employer Portal", response.data)

    def test_job_seeker_route_get(self):
        response = self.app.get('/job_seeker')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Job Seeker Portal", response.data)

    def test_listing_route(self):
        response = self.app.get('/listings')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Current Job Listings", response.data)
        self.assertIn(b"Python Developer", response.data) # Check for mock data

if __name__ == '__main__':
    unittest.main()
