from django.test import TestCase


class TestViews(TestCase):
    def test_call_index_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/index.html')

    def test_call_test_loads(self):
        response = self.client.get('/test')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/test.html')
