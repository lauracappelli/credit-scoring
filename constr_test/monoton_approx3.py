from monoton_approx2 import main
l=0
iterat = 6
for item in range(iterat):
	print("*")
	res = main()
	if res == 1:
		l+=1
print(l/iterat)

