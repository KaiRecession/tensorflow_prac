#_*_coding:utf-8_*_
# 作者      ：liuxiaowei
# 创建时间   ：4/23/22 10:51 PM
# 文件      ：sniff_pcap.py
# IDE      ：PyCharm

from scapy.all import *
import time
import optparse


# 时间戳转换函数
def TimeStamp2Time(timeStamp):
    timeTmp = time.localtime(timeStamp)
    myTime = time.strftime("%Y-%m-%d %H:%M:%S", timeTmp)
    return myTime

# 回调输出函数
def PackCallBack(packet):
     print("=" * 30)
     # 打印源IP，源端口，目的IP，目的端口
     print(f"[{TimeStamp2Time(packet.time)}]Source:{packet[IP].src}:{packet.sport}--->Target:{packet[IP].dst}:{packet.dport}")

     # 打印输出数据包
     print(packet.show())
     print("=" * 30)

if __name__ == '__main__':

    hostIP = input('请输入目的IP地址：')
    fileName = input('请输入保存的文件名：')
    packetCount = input('请输入捕获的数据包总数：')
    defFilter = "dst " + hostIP
    packets = sniff(filter=defFilter, prn=PackCallBack, count=int(packetCount))
    # 保存输出文件
    wrpcap(fileName, packets)
