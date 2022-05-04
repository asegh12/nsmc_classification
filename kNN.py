def read_train(filename):
    labels = []
    tfidf = []
    with open(filename, "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            splited_line = line.split(" ")
            if len(splited_line) == 1 : continue
            tfidf.append({})
            labels.append(int(splited_line[0]))
            for pair in splited_line[1:]:
                if pair == '\n' : continue
                n, v = pair.split(":")
                tfidf[-1][float(n)] = float(v)
    return labels, tfidf


if __name__ == "__main__":
    # 문서벡터 가져오기(라벨값과 각 문서의 tfidf)
    label_trains, tfidf_train = read_train("tfidf_train.txt")
    cnt = 0
    # 평가할 테스트 데이터 문장 개수
    for_cnt = 1000
    
    tp = 0
    fp = 0
    fn = 0
    tn  = 0
    
    with open('tfidf_test.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
        
        # 테스트 데이터 평가    
        for i, line in zip(range(50000), lines):
            print(i)
            if for_cnt == i : break
            cos_sim_list = []
            splited_line = line.split(" ")
            if len(splited_line) == 1 : continue
            test_label = int(splited_line[0])
            
            # 테스트 데이터 라벨과 각 문서 tfidf 가져오기
            test_tfidf = []
            for pair in splited_line[1:]:
                if pair == '\n' : continue
                n, v = pair.split(":")
                test_tfidf.append((float(n), float(v)))
                
            # 유사도 검사
            for line_tfidf, label in zip(tfidf_train, label_trains):
                cos_sim = 0
                for n, v in test_tfidf:
                    if n in line_tfidf : cos_sim += v * line_tfidf[n]
                
                cos_sim_list.append((cos_sim, label))
            
            # 상위 5개 문장으로 분류
            topn = sorted(cos_sim_list, key = lambda x: -x[0])[:5]
            
            pos = 0
            
            for c in topn:
                if c[1] == 1 : pos += 1
            predict = 1 if pos >= 3 else -1
            if predict == test_label : 
                cnt += 1
                if predict==1 : tp += 1
                else : tn += 1
            else :
                if predict==1 : fp +=1
                else : fn +=1
            
    print(f'Accuracy : {cnt/for_cnt}')
    print(f'Precision : {tp/(tp+fp)}')
    print(f'Recall : {tp/(tp+fn)}')
    