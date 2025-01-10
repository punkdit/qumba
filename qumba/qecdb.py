#!/usr/bin/env python

"""
to run:
python -m flask --app qecdb run --debug

"""

from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

main_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {font-family: Arial, Helvetica, sans-serif;}
form {border: 3px solid #f1f1f1;}

input[type=text], input[type=password] {
  width: 100%;
  padding: 12px 20px;
  margin: 8px 0;
  display: inline-block;
  border: 1px solid #ccc;
  box-sizing: border-box;
}

button {
  background-color: #04AA6D;
  color: white;
  padding: 14px 20px;
  margin: 8px 0;
  border: none;
  cursor: pointer;
  width: 100%;
}

button:hover {
  opacity: 0.8;
}

.cancelbtn {
  width: auto;
  padding: 10px 18px;
  background-color: #f44336;
}

.imgcontainer {
  text-align: center;
  margin: 24px 0 12px 0;
}

img.avatar {
  width: 40%;
  border-radius: 50%;
}

.container {
  padding: 16px;
}

span.psw {
  float: right;
  padding-top: 16px;
}

/* Change styles for span and cancel button on extra small screens */
@media screen and (max-width: 300px) {
  span.psw {
     display: block;
     float: none;
  }
  .cancelbtn {
     width: 100%;
  }
}
</style>
</head>
<body>

<form action="/search" method="post">
  <div class="container">
    <label for="username"><b>name</b></label>
    <input type="text" placeholder="eg. [[16,4,4]]" name="name">

    <button type="submit">Search</button>
  </div>
</form>

RESULTS

</body>
</html>
"""

@app.route("/")
def main():
    return main_html.replace("RESULTS", "")


from qumba import db

def parse_name(name):
    for c in " []":
        name = name.replace(c, "")
    flds = name.split(",")
    assert len(flds) == 3
    n, k, d = flds
    n = int(n)
    k = int(k)
    d = int(d)
    name = "[[%s, %s, %s]]" % (n, k, d)
    return name


@app.route('/search', methods=['POST', 'GET'])
def search():
    error = None
    if request.method != 'POST':
        return main_html.replace("RESULTS", "")

    query = {}

    name = request.form["name"]
    if name:
        name = parse_name(name)
        query["name"] = name

    res = db.codes.find_one(query)

    results = "<p> %s </p>"%res

    return main_html.replace("RESULTS", results)




