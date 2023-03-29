

with open(r'C:\Users\i_hat\Desktop\soy.txt', 'r', encoding = 'utf-8') as f:
    copy = f.read()



done = set()
result = ''
lines = [i + '\n' for i in copy.split('\n')]

for line in lines:
    if line.strip() == '':
        result = result + line 
        continue
    if line not in done:
        result = result + line
        done.add(line)

print(result)

with open(r'C:\Users\i_hat\Desktop\soy2.txt', 'w', encoding = 'utf-8') as f:
    f.write(result)
