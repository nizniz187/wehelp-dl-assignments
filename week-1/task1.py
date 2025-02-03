import urllib.request
import ssl
import json

URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31"
OUTPUT_FILE_NAME = 'products.txt'

print('Task 1 started.')

def get_prod_id_list():
  # Fetch data
  print(f'  > Parsing page {page}...')
  req = urllib.request.Request(f'{URL}&page={page}')
  with urllib.request.urlopen(req, context=context) as response:
    data = response.read().decode("utf-8")
  # Parse data to product ids
  json_data = json.loads(data)
  prods = json_data.get('Prods')
  return list(map(lambda x: x.get('Id'), prods))

context = ssl._create_unverified_context()
page = 1
prod_id_list = []
while page > 0:
  ids = get_prod_id_list()
  if len(ids) == 0:
    break
  else:
    prod_id_list += ids
    page += 1

# Output file
with open(OUTPUT_FILE_NAME, "w") as file:
  for id in prod_id_list:
      file.write(f'{id}\n')

print(f'Task 1 done. Output file: {OUTPUT_FILE_NAME}.')