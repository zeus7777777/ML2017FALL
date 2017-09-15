import sys

with open(sys.argv[1]) as f:
    ss = f.read().split()
    #print(ss)
    word = []
    word_cnt = []
    for s in ss:
        f = 0
        idx = 0
        for i in range(len(word)):
            if word[i]==s:
                f = 1
                idx = i
                break
        if f==1:
            word_cnt[idx] += 1
        else:
            word.append(s)
            word_cnt.append(1)
    with open('Q1.txt', 'w') as g:
        for i in range(len(word)):
            g.write(word[i]+' '+str(i)+' '+str(word_cnt[i]))
            if i<len(word)-1:
                g.write('\n')