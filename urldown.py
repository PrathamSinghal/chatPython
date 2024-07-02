import requests

# Making a get request
query_parameters = {"downloadformat": "pdf"}
response = requests.get('https://app.box.com/shared/static/xb03u32krysahuilj0kdf4yho4cn0ca6.pdf', params=query_parameters)

# print response
print(response)

# print url
print(response.content)
pdf = open("pdf"+"x"+".pdf", 'wb')
pdf.write(response.content)
pdf.close()