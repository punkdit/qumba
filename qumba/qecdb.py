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

body_html = """
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

the_header = """
<h2>QEC database</h2>
"""

at_symbol = """
<svg fill="#000000" height="10px" width="10px" version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
	 viewBox="0 0 378.632 378.632" xml:space="preserve">
<path d="M377.406,160.981c-5.083-48.911-31.093-92.52-73.184-122.854C259.004,5.538,200.457-6.936,147.603,4.807
	C97.354,15.971,53.256,48.312,26.571,93.491C-0.122,138.731-7.098,192.982,7.436,242.39c7.832,26.66,21.729,51.712,40.15,72.51
	c18.594,20.972,41.904,37.722,67.472,48.459c23.579,9.888,48.628,14.797,73.653,14.797c34.128-0.001,68.115-9.121,97.949-27.098
	l-21.092-35.081c-40.578,24.451-90.887,28.029-134.652,9.66c-40.283-16.96-71.759-52.383-84.211-94.761
	c-11.336-38.595-5.846-81.093,15.125-116.586c20.922-35.467,55.426-60.801,94.622-69.533c41.644-9.225,87.948,0.669,123.857,26.566
	c32.502,23.394,52.497,56.769,56.363,93.907c2.515,23.979,0.31,42.891-6.526,56.226c-14.487,28.192-35.526,28.36-43.873,27.132
	c-0.283-0.041-0.476-0.082-0.65-0.117c-2.396-3.709-2.091-17.489-1.974-23.473c0.044-2.332,0.084-4.572,0.084-6.664v-112.06h-31.349
	c-3.998-3.278-8.225-6.251-12.674-8.921c-17.076-10.159-36.858-15.552-57.255-15.552c-29.078,0-56.408,10.597-76.896,29.824
	c-32.537,30.543-42.63,80.689-24.551,122.023c8.578,19.62,23.065,35.901,41.876,47.066c17.611,10.434,38.182,15.972,59.47,15.972
	c24.394,0,46.819-6.735,64.858-19.492c1.915-1.342,3.813-2.79,5.626-4.233c6.431,8.805,15.811,14.4,27.464,16.114
	c16.149,2.408,32.299-0.259,46.784-7.668c16.453-8.419,29.715-22.311,39.439-41.271C377.209,219.346,380.778,193.46,377.406,160.981
	z M242.33,224.538c-0.891,1.283-2.229,2.907-2.961,3.803c-0.599,0.778-1.151,1.46-1.643,2.073
	c-3.868,4.982-8.597,9.48-14.113,13.374c-11.26,7.943-25.152,11.964-41.257,11.964c-28.968,0-53.462-14.75-63.846-38.544
	c-11.258-25.69-5.071-56.854,15.035-75.692c12.7-11.95,30.538-18.784,48.911-18.784c13.028,0,25.56,3.375,36.268,9.788
	c6.831,4.072,12.861,9.337,17.9,15.719c0.497,0.613,1.082,1.322,1.724,2.094c0.952,1.135,2.812,3.438,3.981,5.092V224.538z"/>
</svg>
"""


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
    def __init__(self, longname, name, items):
        self.items = [""] + items.split("/")
        self.longname = longname
        self.name = name
    def render(self, form={}):
        value = form.get(self.name)
        #print("Select:", value)
        items = ' '.join('<option value="%s" %s>%s</option>' % (
            item, "selected" if item==value else "", item) 
            for item in self.items)
        s = '<b>%s:</b><select name="%s">%s</select>' % (
            self.longname,
            self.name,
            items)
        return s
class Para:
    def render(self,form={}):
        return "<p>"
class EndPara:
    def render(self,form={}):
        return "</p>"
class SearchButton:
    def render(self,form={}):
        return '<button type="submit">Search</button>'
    

layout = [
    #Para(),
        #Input("name", "name", 10, "eg. [[17,1,5]]"),
        Input("n",    "n",     6, "eg. 12-20"),
        Input("k",    "k",     6, ),
        Input("d",    "d",     6, "eg. >=4"),
    #EndPara(),
    #Para(),
        #Checkbox("css", "css"),
        #Checkbox("gf4", "gf4"),
        #Checkbox("self-dual", "selfdual"),
        Select("type", "tp", "css/gf4/selfdual"),
        Select("family", "desc", 
    "toric/2BGA/codetables/triorthogonal/CSS-T/bivariate bicycle/hypergraph_product/hyperbolic_2d"),
    #EndPara(),
    SearchButton(),
]

def html_form(form={}):
    items = [f.render(form) for f in layout]
    #items.append('<p><button type="submit">Search</button></p>')
    s = '\n'.join(items)
    s = """
    <form action="../codes/" method="get">
    %s
    </form>
    """ %s
    return s

from qumba import db
from bson.objectid import ObjectId
the_count = db.codes.count_documents({})

the_email = """
<div align="right">admin%sqecdb.org</div>
""" % at_symbol


def html_search(results="", form={}):
    html = the_header
    html += "<div>codes in db: %s</div>"%the_count
    html += the_email
    html += html_form(form)
    html += results
    html = body_html.replace("BODY", html)
    return html


@app.route("/")
def main():
    #return html_search(form={"css":"on"})
    html = the_header
    html += '<a href="https://qecdb.org/codes/">codes</a>'
    html += the_email
    #html += '<a href="codes">codes</a>' # sends to http://127.0.0.1:5000/codes/
    html = body_html.replace("BODY", html)
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
        #return html_search(form={"css":"on"})
        return html_search(form={})

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

#    css = form.get("css")
#    gf4 = form.get("gf4")
#    selfdual = form.get("selfdual")
#
#    if css: query["css"] = True
#    if gf4: query["gf4"] = True
#    if selfdual: query["selfdual"] = True

    tp = form.get("tp")
    if tp == "gf4":
        query["gf4"] = True
    if tp == "css":
        query["css"] = True
    if tp == "selfdual":
        query["selfdual"] = True

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
    result = db.codes.find_one({"_id":ObjectId(_id)})

    if result is None:
        return body_html.replace("BODY", "code %s not found"%_id)

    r = '<a href="../codes/">start again...</a>'
    r += "<h2>Code:</h2>"

    keys = list(result.keys())
    skeys = """
        name n k d d_lower_bound d_upper_bound dx dz desc created
        cssname css gf4 selfdual tp G homogeneous shape 
    """.strip().split()
    keys.sort(key = lambda k : (skeys.index(k) if k in skeys else 999))
    for key in "H T L _id".split():
        if key in keys:
            keys.remove(key)
            keys.append(key)

    d = result.get("d")
    d_lower_bound = result.get("d_lower_bound")
    d_upper_bound = result.get("d_upper_bound")
    dx = result.get("dx")
    dz = result.get("dz")

    if d is not None:
        if d_lower_bound == d:
            keys.remove("d_lower_bound")
        if d_upper_bound == d:
            keys.remove("d_upper_bound")

    rows = []
    for key in keys:
        #if key == "_id":
        #    continue
        value = result[key]
        if key in "HTL":
            value = "<tt>%s</tt>"%value.replace(" ", "<br>")
        elif key == "created":
            value = datetime.fromtimestamp(value)
        if key == "desc":
            key = "family" # TODO fix in db
        tds = '<td>%s</td> <td>%s</td>' % (key, value)
        rows.append("<tr> %s </tr>" % tds)
    r += "<table>%s</table>"%("\n".join(rows),)
    r = "<p> %s </p>"%r

    html = body_html.replace("BODY", r)

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






