with open('main.py', 'r') as file:
    content = file.read()

content = content.replace("@app.route('/entity/<n>')", "@app.route('/entity/<name>')")

with open('main.py', 'w') as file:
    file.write(content)
