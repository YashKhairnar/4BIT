from selenium import webdriver
from selenium.webdriver.common.by import By
import requests

driver = webdriver.Chrome()
driver.get('http://127.0.0.1:5000/predictiveanalysis')
print(driver.title)