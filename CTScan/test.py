from selenium import webdriver
from selenium.webdriver.common.by import By
import requests

driver = webdriver.Chrome()
driver.get('http://127.0.0.1:5000')
print(driver.title)

### 
anchors = driver.find_elements(By.TAG_NAME,'a')
links = []
for anchor in anchors:
    href = anchor.get_attribute('href')
    links.append(href)
print(links)

for link in links:
    response = requests.head(link)
    if response.status_code==200:
        print(link, "[Status 200, OK !]")
    else:
        print(link, "[Not working ! ]")
        
driver.quit()