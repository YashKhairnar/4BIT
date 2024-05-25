import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

plasma_list = ['asparagine', 'benzoic acid', 'tryptophan', 'uric acid', '5-hydroxynorvaline NIST',
               'alpha-ketoglutarate', 'citrulline', 'glutamine', 'hypoxanthine', 'malic acid',
               'methionine sulfoxide', 'nornicotine', 'octadecanol', '3-phosphoglycerate', 
               '5-methoxytryptamine', 'adenosine-5-monophosphate', 'aspartic acid', 'lactic acid', 
               'maltose', 'maltotriose', 'N-methylalanine', 'phenol', 'phosphoethanolamine', 
               'pyrophosphate', 'pyruvic acid', 'taurine']
serum_list = ['cholesterol', 'lactic acid', 'N-methylalanine', 'phenylalanine', 'aspartic acid', 
              'deoxypentitol', 'glutamic acid', 'malic acid', 'phenol', 'taurine']

class FlaskAppTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        chromedriver_path = "C:\\Users\\athar\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
        chrome_options = Options()
        service = Service(chromedriver_path)
        cls.driver = webdriver.Chrome(service=service, options=chrome_options)
        cls.driver.maximize_window()
        cls.driver.implicitly_wait(30)
        cls.driver.set_page_load_timeout(50)

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_home_page(self):
        self.driver.get("http://127.0.0.1:5000/")
        time.sleep(2)
        self.assertIn("Every breath is a gift", self.driver.page_source)


    def test_about_page(self):
        self.driver.get("http://127.0.0.1:5000/")
        time.sleep(2)
        self.driver.find_element(By.LINK_TEXT, "About").click()
        time.sleep(2)
        self.assertIn("About", self.driver.page_source)

    def test_upload_image(self):
        self.driver.get("http://127.0.0.1:5000/")
        time.sleep(2)
        self.driver.find_element(By.LINK_TEXT, "CTScan Analysis").click()
        time.sleep(2)
        
        # Locate the file input element and send the file path to it
        upload_button = self.driver.find_element(By.NAME, "myfile")
        upload_button.send_keys("D:\\GitHub\\4BIT\\CTScan\\Dataset\\test\\000108 (6).png")
        
        # Click the submit button
        submit_button = self.driver.find_element(By.CSS_SELECTOR, ".btn-submit")
        submit_button.click()
        
        # Wait for the result to be generated and check the result in the page source
        time.sleep(30)
        self.assertIn("result", self.driver.page_source)


    def test_metabolite_analysis(self):
        self.driver.get("http://127.0.0.1:5000/")
        time.sleep(2)
        self.driver.find_element(By.LINK_TEXT, "Metabolite Analysis").click()
        time.sleep(2)
        self.assertIn("Metabolite Analysis", self.driver.page_source)

        for i in range(len(plasma_list)):
            field = self.driver.find_element(By.NAME, f"plasma_{i}")
            field.send_keys("1.0")

        for i in range(len(serum_list)):
            field = self.driver.find_element(By.NAME, f"serum_{i}")
            field.send_keys("1.0")

        self.driver.find_element(By.XPATH, "//input[@type='submit']").click()
        time.sleep(5)
        self.assertIn("Lung Cancer Detected", self.driver.page_source)

if __name__ == "__main__":
    unittest.main()
