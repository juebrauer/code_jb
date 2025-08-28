Here is some helpful Linux knowledge


# tree: only show 5 files per folder

<pre>
find . -type d -print | while read -r d; do
  printf '%s\n' "$d"
  find "$d" -maxdepth 1 -type f -printf '%p\n' | sort | head -n 5
done | tree --fromfile .
</pre>