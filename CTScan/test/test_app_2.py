import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        # Setup Chrome driver
        self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        self.driver.get("http://127.0.0.1:5000/")  # Adjust the URL as necessary for your Flask app

    def test_home_page(self):
        driver = self.driver
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        home_text = driver.find_element(By.TAG_NAME, "h1").text
        self.assertEqual(home_text, "Welcome to Lung Cancer Detection")

    def test_about_page(self):
        driver = self.driver
        driver.find_element(By.LINK_TEXT, "About").click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        self.assertIn("About", driver.title)
        about_text = driver.find_element(By.TAG_NAME, "h1").text
        self.assertEqual(about_text, "About Us")
        
    def test_metaboliteanalysis_page(self):
        driver = self.driver
        driver.find_element(By.LINK_TEXT, "Metabolite Analysis").click()
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        analysis_text = driver.find_element(By.TAG_NAME, "h1").text
        self.assertEqual(analysis_text, "Metabolite Analysis")
        # Assuming there are elements or text that mention plasma and serum list
        self.assertIsNotNone(driver.find_element(By.ID, "plasma_list"))
        self.assertIsNotNone(driver.find_element(By.ID, "serum_list"))

    def tearDown(self):
        self.driver.close()

if __name__ == "__main__":
    unittest.main()
