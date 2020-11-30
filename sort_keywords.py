with open('default_keywords.txt', 'r') as keyword_file:
    text = keyword_file.read()

lines = text.split('\n')
lines = [line.strip() for line in lines if len(line)>1]
lines = sorted(lines)
for i in lines:
    print(i)

with open('default_keywords.txt', 'w') as keyword_file:
    keyword_file.write('\n'.join(lines))
