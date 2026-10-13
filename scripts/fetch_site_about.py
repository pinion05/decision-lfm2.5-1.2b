import urllib.request, html, re
t = urllib.request.urlopen("https://zefan-cai.github.io/open-jev/", timeout=30).read().decode("utf-8", "ignore")
t = re.sub("<script.*?</script>", " ", t, flags=re.S)
t = re.sub("<style.*?</style>", " ", t, flags=re.S)
t = re.sub("<[^>]+>", " ", t)
t = html.unescape(t)
t = re.sub(r"\s+", " ", t)
i = t.lower().find("background")
print(t[:900] if i < 0 else t[max(0, i - 200):i + 900])
