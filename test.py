import medscan

text = 'token1: some arbitrary text; token1 token2: some more really arbitrary text; token1: unless....;'
for i in range(3):
    print('-----------')
print('Initial text:')
print(text)
for i in range(3):
    print('-----------')
result = medscan.chinchoppa(text)#, keywords=['token1', 'token1 token2'])
print(result)


