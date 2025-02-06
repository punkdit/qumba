#!/usr/bin/env python

"""
to run:
python -m flask --app qecdb run --debug

"""

from datetime import datetime

is_dev = True

from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

css = """
<style>
body {font-family: Arial, Helvetica, sans-serif;}
form {border: 3px solid #f1f1f1;}

input {
  padding: 4px 4px;
  margin: 8px 0;
  display: inline-block;
  border: 1px solid #ccc;
  box-sizing: border-box;
}

button {
  background-color: #04AA6D;
  color: white;
  padding: 6px 6px;
  margin: 6px 0;
  border: none;
  cursor: pointer;
  width: 100px;
}

button:hover {
  opacity: 0.8;
}

table {
  border-collapse: collapse;
  border: 1px solid;
}

td {
  border: 1px solid;
  padding: 5px;
}

tr {
  padding: 5px;
}

</style>
"""

main_html = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
%s
</head>
<body>
BODY
</body>
</html>
""" % css

class Input:
    def __init__(self, longname, name, size=None, 
            placeholder=None, value=None, checked=False):
        self.longname = longname
        self.name = name
        self.size = size
        self.placeholder = placeholder
        self.value = value
        self.checked = checked
    def render(self, form={}):
        args = 'type="text" name="%s" ' % (self.name,)
        if self.size: 
            args += 'size="%s" '%self.size
        if self.placeholder: 
            args += 'placeholder="%s" '%self.placeholder
        value = form.get(self.name)
        if value is not None:
            args += 'value="%s" '%value
        elif self.value: 
            args += 'value="%s" '%self.value
        if self.checked: 
            args += ' checked '
        s = "<b>%s:</b><input %s />"%(self.longname, args)
        return s

class Checkbox(Input):
    def render(self, form={}):
        args = 'type="checkbox" name="%s" ' % (self.name,)
        value = form.get(self.name)
        #print("Checkbox", self.name, value)
        if value=="on" or self.checked: 
            args += ' checked '
        s = "<b>%s:</b><input %s />"%(self.longname, args)
        return s

class Select(Input):
    def __init__(self, name, items):
        self.items = [""] + items.split("/")
        self.name = name
    def render(self, form={}):
        value = form.get(self.name)
        print("Select:", value)
        items = ' '.join('<option value="%s" %s>%s</option>' % (
            item, "selected" if item==value else "", item) 
            for item in self.items)
        s = '<b>%s:</b><select name="%s">%s</select>' % (
            self.name,
            self.name,
            items)
        return s
    

layout = [
    #Input("name", "name", 10, "eg. [[17,1,5]]"),
    Input("n",    "n",     6, "eg. 12-20"),
    Input("k",    "k",     6, ),
    Input("d",    "d",     6, "eg. >=4"),
    Checkbox("css", "css"),
    Checkbox("gf4", "gf4"),
    Checkbox("self-dual", "selfdual"),
    Select("desc", 
    "toric/2BGA/codetables/triorthogonal/CSS-T/bivariate bicycle/hypergraph_product/hyperbolic_2d"),
]

def html_form(form={}):
    s = '\n'.join(f.render(form) for f in layout)
    s += '\n<p><button type="submit">Search</button></p>'
    s = """
    <form action="../codes/" method="get">
    %s
    </form>
    """ %s
    return s

from qumba import db
from bson.objectid import ObjectId
the_count = db.codes.count_documents({})


def html_search(results="", form={}):
    html = "<h2>QEC database</h2>"
    html += "<p>codes in db: %s</p>"%the_count
    html += html_form(form)
    html += results
    html = main_html.replace("BODY", html)
    return html


@app.route("/")
def main():
    #return html_search(form={"css":"on"})
    html = "<h2>QEC database</h2>"
    #html += '<a href="https://qecdb.org/codes/">codes</a>'
    html += '<a href="codes">codes</a>'
    html = main_html.replace("BODY", html)
    return html


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

@app.route('/codes/', methods=['POST', 'GET'])
def codes():
    error = None
    #if request.method != 'POST':
    #    return html_search(form={"css":"on"})

    query = {}

#    name = request.form["name"]
#    if name:
#        name = parse_name(name)
#        query["name"] = name

    print("request.form:", request.form)
    print("request.args:", request.args)
    form = request.args

    if request.method == "POST" or not form:
        return html_search(form={"css":"on"})

    for attr in "nkd":
        value = form.get(attr, "")
        value = value.strip()
        if not value:
            continue
        try:
            query[attr] = int(value)
            continue
        except ValueError:
            pass
        if "-" in value:
            lhs, rhs = value.split("-")
            value = {}
            if lhs: value["$gte"] = int(lhs)
            if rhs: value["$lte"] = int(rhs)
            query[attr] = value
            continue
        ops = {">=":"$gte", ">":"$gt", "<":"$lt", "<=":"$lte"}
        for op in ">= > <= <".split():
            if op in value:
                value = value.replace(op, "")
                value = int(value)
                value = {ops[op]:value}
                query[attr] = value
                break
        else:
            assert 0, "wup: %s"%value

    css = form.get("css")
    gf4 = form.get("gf4")
    selfdual = form.get("selfdual")

    if css: query["css"] = True
    if gf4: query["gf4"] = True
    if selfdual: query["selfdual"] = True

    desc = form.get("desc")
    if desc:
        query["desc"] = desc

    print(query)

    #res = db.codes.find_one(query)
    count = db.codes.count_documents(query)
    cursor = db.codes.find(query)
    cursor.sort("n")
    limit = 100
    cursor = cursor[:limit]

    if count>1:
        r = "<p>%s codes found</p>"%count
    elif count == 1:
        r = "<p>1 code found</p>"
    else:
        r = "<p>no codes found</p>"

    if count > limit:
        r += "<p>showing first %s:</p>"%limit

    items = list(cursor)
    def fn(item):
        n,k,d = item["n"], item["k"], item["d"]
        tp = item.get("tp")
        return n,k,d,tp
    items.sort(key = fn)

    COLS = 4

    rows = []
    idx = 0
    row = []
    for item in items:
        _id = item["_id"]
        name = item["name"]
        tp = item.get("tp", "")
        tp = tp if tp!="none" else ""
        td = '<td><a href="%s">%s</a> %s</td>'%("/codes/%s"%_id, name, tp)
        row.append(td)
        if len(row) == COLS:
            rows.append("<tr> %s </tr>" % " ".join(row))
            row = []
    if row:
        rows.append("<tr> %s </tr>" % " ".join(row))
    r += "<table>%s</table>"%("\n".join(rows),)
    r = "<p> %s </p>"%r

    s = html_search(r, form)

    #print(s)

    return s



@app.route("/codes/<_id>")
def codes_id(_id):
    res = db.codes.find_one({"_id":ObjectId(_id)})

    if res is None:
        return main_html.replace("BODY", "code %s not found"%_id)

    r = '<a href="../codes/">start again...</a>'
    r += "<h2>Code:</h2>"

    keys = list(res.keys())
    skeys = """
        name n k d d_lower_bound d_upper_bound dx dz desc created
        cssname css gf4 selfdual tp H T L _id
    """.strip().split()
    keys.sort(key = lambda k : (skeys.index(k) if k in skeys else 999))

    rows = []
    for key in keys:
        #if key == "_id":
        #    continue
        value = res[key]
        if key in "HTL":
            value = "<tt>%s</tt>"%value.replace(" ", "<br>")
        elif key == "created":
            value = datetime.fromtimestamp(value)
        tds = '<td>%s</td> <td>%s</td>' % (key, value)
        rows.append("<tr> %s </tr>" % tds)
    r += "<table>%s</table>"%("\n".join(rows),)
    r = "<p> %s </p>"%r

    html = main_html.replace("BODY", r)

    #print(html)

    return html


if __name__=="__main__":
    # do we need this?
    #from werkzeug.middleware.proxy_fix import ProxyFix
    #app.wsgi_app = ProxyFix( app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    is_dev = False

    #app.run()
    import eventlet
    from eventlet import wsgi
    wsgi.server(eventlet.listen(("127.0.0.1", 5000)), app)






