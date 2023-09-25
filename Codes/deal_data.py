
"""
split train Datasets and  ind_test datasest 7:3
"""
from sklearn.model_selection import train_test_split

def deal_data(filename,isPositive=True):

    with open(file=filename,mode='r',encoding='utf-8') as f:
        i=1
        seq_list=[]
        label_list = []
        for line in f.readlines():
            if i % 2!=0:
                if isPositive is True:
                    label_list.append('1')
                else:
                    label_list.append('0')
                i+=1
            else:
                # print(line)
                seq_list.append(line.strip())
                i+=1
                # print(seq_list)
        f.close()
        print("处理数据完毕")
        return seq_list,label_list


"""
处理数据，将数据转为csv的文件格式：
"""
def get_data(seq_list,label_list,save_train_filepath,save_test_path):
    # 进行7 ：3划分数据集
    # with open(save_filepath,mode='w+',encoding='utf-8') as f:

    X=seq_list
    y=label_list

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

    with open(save_train_filepath,mode='w+',encoding='utf-8') as f1:

        for sequence,lable in zip(X_train,y_train):
            f1.write(sequence +',' +lable+'\n')

    f1.close()

    with open(save_test_path, mode='w+', encoding='utf-8') as f2:

        for sequence, lable in zip(X_test, y_test):
            f2.write(sequence + ',' + lable + '\n')

    f2.close()
    print("转成csv文件完成！")

if __name__ == '__main__':



    # CommonWheats_Pos_data_filepath = '../Datasets/fasta_40/commonWheatPos_40.fasta'
    # CommonWheats_Neg_data_filepath = '../Datasets/fasta_40/commonWheatNeg_40.fasta'

    # 小麦正样本：
    # commonWheats_seq_Pos_list,commonWheats_label_Pos_list=deal_data(CommonWheats_Pos_data_filepath,isPositive=True)
    # get_data(commonWheats_seq_Pos_list,commonWheats_label_Pos_list,save_train_filepath="../Csv/commonWheats_train_Post.Csv",save_test_path="../Csv/commonWheats_test_Pos.Csv")
    # #小麦负样本：
    # commonWheats_seq_Neg_list, commonWheats_label_Neg_list = deal_data(CommonWheats_Neg_data_filepath, isPositive=False)
    # get_data(commonWheats_seq_Neg_list, commonWheats_label_Neg_list, save_train_filepath="../Csv/commonWheats_train_Neg.Csv",save_test_path="../Csv/commonWheats_test_Neg.Csv")



    # wheatSeeding_Pos_data_filepath="../Datasets/fasta_40/wheatSeedingPos_40.fasta"
    # wheatSeeding_Neg_data_filepath="../Datasets/fasta_40/wheatSeedingNeg_40.fasta"

    # wheatSeeding_seq_Pos_list, wheatSeedinglabel_Pos_list = deal_data(wheatSeeding_Pos_data_filepath, isPositive=True)
    # get_data(wheatSeeding_seq_Pos_list, wheatSeedinglabel_Pos_list, save_train_filepath="../Csv/wheatSeeding_train_Pos.Csv",
    #          save_test_path="../Csv/wheatSeeding_test_Pos.Csv")

    # wheatSeeding_seq_Neg_list, wheatSeeding_label_Neg_list = deal_data(wheatSeeding_Neg_data_filepath, isPositive=False)
    # get_data(wheatSeeding_seq_Neg_list, wheatSeeding_label_Neg_list, save_train_filepath="../Csv/wheatSeeding_train_Neg.Csv",
    #          save_test_path="../Csv/wheatSeeding_test_Neg.Csv")

    # Rice_Pos_data_filepath = '../Datasets/fasta_40/ricePos_40.fasta'
    # Rice_Neg_data_filepath = '../Datasets/fasta_40/riceNeg_40.fasta'
    # # 水稻正样本：
    # rice_seq_Pos_list, rice_label_Pos_list = deal_data(Rice_Pos_data_filepath, isPositive=True)
    # get_data(rice_seq_Pos_list, rice_label_Pos_list, save_train_filepath="../Csv/rice_train_Pos.Csv", save_test_path="../Csv/rice_test_Pos.Csv")
    # # 水稻负样本：
    # rice_seq_Neg_list, rice_label_Neg_list = deal_data(Rice_Neg_data_filepath, isPositive=False)
    # get_data(rice_seq_Neg_list, rice_label_Neg_list, save_train_filepath="../Csv/rice_train_Neg.Csv", save_test_path="../Csv/rice_test_Neg.Csv")
    #
    #
    # Tabacum_Pos_data_filepath = '../Datasets/fasta_40/tabacumPos_40.fasta'
    # Tabacum_Neg_data_filepath = '../Datasets/fasta_40/tabacumNeg_40.fasta'

    # 烟草positive：

    # tabacum_seq_Pos_list, tabacum_label_Pos_list = deal_data(Tabacum_Pos_data_filepath, isPositive=True)
    # get_data(tabacum_seq_Pos_list, tabacum_label_Pos_list, save_train_filepath="../Csv/tabacum_train_Pos.Csv",save_test_path="../Csv/tabacum_test_Pos.Csv")


    # 烟草negative：
    # tabacum_seq_Neg_list, tabacum_label_Neg_list = deal_data(Tabacum_Neg_data_filepath, isPositive=False)
    # get_data(tabacum_seq_Neg_list, tabacum_label_Neg_list, save_train_filepath="../Csv/tabacum_train_Neg.Csv", save_test_path="../Csv/tabacum_test_Neg.Csv")




    #peanut
    peanut_Pos_data_filepath = "../Datasets/fasta_40/peanutPos_40.fasta"
    peanut_Neg_data_filepath = "../Datasets/fasta_40/peanutNeg_40.fasta"

    peanut_seq_Pos_list, peanut_label_Pos_list = deal_data(peanut_Pos_data_filepath, isPositive=True)
    get_data(peanut_seq_Pos_list, peanut_label_Pos_list,
             save_train_filepath="../Csv/peanut_train_Pos.Csv",
             save_test_path="../Csv/peanut_test_Pos.Csv")

    peanut_seq_Neg_list, peanut_label_Neg_list = deal_data(peanut_Neg_data_filepath, isPositive=False)
    get_data(peanut_seq_Neg_list, peanut_label_Neg_list,
             save_train_filepath="../Csv/peanut_train_Neg.Csv",
             save_test_path="../Csv/peanut_test_Neg.Csv")



    #papaya
    papaya_Pos_data_filepath = "../Datasets/fasta_40/papayaPos_40.fasta"
    papaya_Neg_data_filepath = "../Datasets/fasta_40/papayaNeg_40.fasta"

    papaya_seq_Pos_list, papaya_label_Pos_list = deal_data(papaya_Pos_data_filepath, isPositive=True)
    get_data(papaya_seq_Pos_list, papaya_label_Pos_list,
             save_train_filepath="../Csv/papaya_train_Pos.Csv",
             save_test_path="../Csv/papaya_test_Pos.Csv")


    papaya_seq_Neg_list, papaya_label_Neg_list = deal_data(papaya_Neg_data_filepath, isPositive=False)
    get_data(papaya_seq_Neg_list, papaya_label_Neg_list,
             save_train_filepath="../Csv/papaya_train_Neg.Csv",
             save_test_path="../Csv/papaya_test_Neg.Csv")

