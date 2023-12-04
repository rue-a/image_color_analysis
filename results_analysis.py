# %%
import json

with open("results.json", "r") as file:
    meta = json.load(file)


tt = 0
tf = 0
ft = 0
ff = 0
for key in meta.keys():
    if meta[key]["color_coded"] and meta[key]["hue_analysis"]["color_coded"]:
        tt += 1
    if meta[key]["color_coded"] and not meta[key]["hue_analysis"]["color_coded"]:
        tf += 1
    if not meta[key]["color_coded"] and meta[key]["hue_analysis"]["color_coded"]:
        ft += 1
    if not meta[key]["color_coded"] and not meta[key]["hue_analysis"]["color_coded"]:
        ff += 1

print("HSV")
print("true true: ", tt)
print("false false: ", ff)
print("true false: ", tf)
print("false true: ", ft)
print()


tt = 0
tf = 0
ft = 0
ff = 0
for key in meta.keys():
    if meta[key]["color_coded"] and meta[key]["lab_analysis"]["color_coded"]:
        tt += 1
        print(meta[key]["lab_analysis"]["hull_area"])

    if meta[key]["color_coded"] and not meta[key]["lab_analysis"]["color_coded"]:
        tf += 1

    if not meta[key]["color_coded"] and meta[key]["lab_analysis"]["color_coded"]:
        ft += 1

    if not meta[key]["color_coded"] and not meta[key]["lab_analysis"]["color_coded"]:
        ff += 1

print("Lab")
print("true true: ", tt)
print("false false: ", ff)
print("true false: ", tf)
print("false true: ", ft)
print()
