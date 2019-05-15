#/usr/bin/python3.6
#coding:utf-8
#五分类测试集数据处理
import numpy as np
import random,csv
import time
import tensorflow as tf
global staut_list
global staut_list0, staut_list1, staut_list2, staut_list3, staut_list4

def preHandle():
    source_file = 'corrected'
    handled_file ='corrected_handled-5-label-1.csv'
    data_to_flie=open(handled_file, 'w',newline='')
    with (open(source_file,'r')) as data_from:
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        count=0
        for i in csv_reader:
            # print i
            temp_line=np.array(i)
            temp_line[1]=handleProtocol(i)         #将源文件行中3种协议类型转换成数字标识
            temp_line[2]=handleService(i)          #将源文件行中70种网络服务类型转换成数字标识
            temp_line[3]=handleFlag(i)             #将源文件行中11种网络连接状态转换成数字标识
            #temp_line[41]=handle5Label(i)           #将源文件行中5大类型转换成数字标识(包含未知攻击)
            temp_line[41] = handle5Label1(i)        # 将源文件行中5大类型转换成数字标识(未知攻击分入一个类型 单独标签)
            csv_writer.writerow(temp_line)
            print(count,'staus:',temp_line[1],temp_line[2],temp_line[3],temp_line[41])
            count+=1
            # return 1
        data_to_flie.close()
def find_index(x,y):
    return [ a for a in range(len(y)) if y[a] == x]
def handleProtocol(input):
    protoclo_list=['tcp','udp','icmp']
    if input[1]in protoclo_list:
        return find_index(input[1],protoclo_list)[0]
def handleService(input):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u','echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames','http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell','smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i','uucp','uucp_path','vmnet','whois','X11','Z39_50']
    if input[2]in service_list:
        return find_index(input[2],service_list)[0]
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3]in flag_list:
        return find_index(input[3],flag_list)[0]

def handleLabel(input):
    global staut_list
    # ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']
    if input[41] in staut_list:
        return find_index(input[41],staut_list)[0]
    else:
        staut_list.append(input[41])
        return find_index(input[41],staut_list)[0]
def handle5Label(input):
    global staut_list
    #normal=0
    staut_list0 = ['normal.']
    # Dos=1//10种
    staut_list1 = ['back.','land.','neptune.','pod.','smurf.','teardrop.','apache2.','mailbomb.','processtable.','udpstorm.']
    # probling=2//6种
    staut_list2 = ['ipsweep.','nmap.','portsweep.','satan.','mscan.','saint.']
    # R2L=3//15种
    staut_list3 = ['ftp_write.','guess_passwd.','imap.','multihop.','phf.','spy.','warezclient.','warezmaster.','named.','sendmail.','snmpgetattack.','snmpguess.','worm.','xlock.','xsnoop.']
    # U2R=4//8种
    staut_list4 = ['buffer_overflow.','loadmodule.','perl.','rootkit.','httptunnel.','ps.','sqlattack.','xterm.']
    # ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']
    if input[41] in staut_list0:
        return 0
    elif input[41]in staut_list1:
        return 1
    elif input[41] in staut_list2:
        return 2
    elif input[41] in staut_list3:
        return 3
    else:
        return 4
def handle5Label1(input):
    global staut_list
    #normal=0
    staut_list0 = ['normal.']
    # Dos=1//10种
    staut_list1 = ['back.','land.','neptune.','pod.','smurf.','teardrop.']
    # probling=2//6种
    staut_list2 = ['ipsweep.','nmap.','portsweep.','satan.']
    # R2L=3//15种
    staut_list3 = ['ftp_write.','guess_passwd.','imap.','multihop.','phf.','spy.','warezclient.','warezmaster.']
    # U2R=4//8种
    staut_list4 = ['buffer_overflow.','loadmodule.','perl.','rootkit.']
    # ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.', 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.', 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.', 'spy.', 'rootkit.']

    staut_list5 =['apache2.', 'mailbomb.', 'processtable.', 'udpstorm.','mscan.','saint.','named.','sendmail.','snmpgetattack.','snmpguess.','worm.','xlock.','xsnoop.','httptunnel.','ps.','sqlattack.','xterm.']
    if input[41] in staut_list0:
        return 0
    elif input[41]in staut_list1:
        return 1
    elif input[41] in staut_list2:
        return 2
    elif input[41] in staut_list3:
        return 3
    elif input[41] in staut_list4:
        return 4
    else:
        return 5

if __name__ == '__main__':
    start_time=time.clock()
    global staut_list0,staut_list1,staut_list2,staut_list3,staut_list4
    staut_list=[]
    preHandle()
    # print staut_list
    end_time=time.clock()
    print("Running time:",(end_time-start_time))  #输出程序运行时间