from selenium import webdriver
from selenium.webdriver.common.by import By
import requests

driver = webdriver.Chrome()
driver.get('http://127.0.0.1:5000/predictiveanalysis')
print(driver.title)

driver.find_element(By.CLASS_NAME,"file").send_keys("C:/Users/athar/Desktop/Dataset/train/cancer/000015.png")
driver.find_element(By.CLASS_NAME, 'btn-submit').submit()
if(driver.page_source.find('File uploaded')):
    print("file upload successfull")
else:
    print("Please uplaod an image")
driver.quit()