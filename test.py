'''num = int(input())
ans = []
for i in range(num):
    a = list(map(int, input().split(' ')))
    planetNum = int(input())
    planetCross = 0
    for j in range(planetNum):
        planetCord = list(map(int, input().split(' ')))
        planetBool1 = (a[0] - planetCord[0]) ** 2 + (a[1] - planetCord[1]) ** 2 > planetCord[2] ** 2
        planetBool2 = (a[2] - planetCord[0]) ** 2 + (a[3] - planetCord[1]) ** 2 > planetCord[2] ** 2
        if planetBool1 != planetBool2:
            planetCross +=1
    ans.append(planetCross)
for i in range(num):
    print(ans[i])'''
