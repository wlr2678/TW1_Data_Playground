# Test with a headless Chrome browser in Selenium to query and obtain POI coordinates from Google Mars
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

CHROME_DRIVER_PATH = r'C:\Selenium\chromedriver.exe'
# initialize the Selenium WebDriver

options = Options()
options.headless = True

driver = webdriver.Chrome(options=options, executable_path=CHROME_DRIVER_PATH)
# visit your target site
driver.get('https://www.google.com/mars/')

# scraping logic...
input()

# release the resources allocated by Selenium
# and shut down the browser
driver.quit()