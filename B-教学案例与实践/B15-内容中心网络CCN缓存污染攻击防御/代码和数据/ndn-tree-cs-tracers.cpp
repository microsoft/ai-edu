/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * Copyright (c) 2011-2015  Regents of the University of California.
 *
 * This file is part of ndnSIM. See AUTHORS for complete list of ndnSIM authors and
 * contributors.
 *
 * ndnSIM is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * ndnSIM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ndnSIM, e.g., in COPYING.md file.  If not, see <http://www.gnu.org/licenses/>.
 **/

// ndn-tree-cs-tracers.cpp

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/ndnSIM-module.h"
#include <cstdio>
namespace ns3 {

/**
 * To run scenario and see what is happening, use the following command:
 *
 *     ./waf --run=ndn-tree-cs-tracers
 */

const char* atk_time="";

int
main(int argc, char* argv[])
{
  CommandLine cmd;
  cmd.Parse(argc, argv);

  // 读取拓扑文件
  AnnotatedTopologyReader topologyReader("", 1);
  topologyReader.SetFileName("src/ndnSIM/examples/topologies/topo-tree-25-node.txt");
  topologyReader.Read();

  ndn::StackHelper ndnHelper;
  // 将所有节点的缓存大小设置为80，缓存替换策略设置为LRU。
  ndnHelper.SetOldContentStore("ns3::ndn::cs::Lru", "MaxSize","80"); 
  ndnHelper.InstallAll();

  // Choosing forwarding strategy
  ndn::StrategyChoiceHelper::InstallAll("/popular", "/localhost/nfd/strategy/best-route");

  ndn::StrategyChoiceHelper::InstallAll("/unpopular", "/localhost/nfd/strategy/best-route");

  // Installing global routing interface on all nodes
  ndn::GlobalRoutingHelper ndnGlobalRoutingHelper;
  ndnGlobalRoutingHelper.InstallAll();

  // Getting containers for the consumer/producer
  Ptr<Node> consumers[16] = {Names::Find<Node>("Src1"), Names::Find<Node>("Src2"),
                            Names::Find<Node>("Src3"), Names::Find<Node>("Src4"),
                            Names::Find<Node>("Src5"), Names::Find<Node>("Src6"),
                            Names::Find<Node>("Src7"), Names::Find<Node>("Src8"),Names::Find<Node>("Src9"),
                            Names::Find<Node>("Dst1"), Names::Find<Node>("Dst2"),
                            Names::Find<Node>("Dst4"), Names::Find<Node>("Dst5"),Names::Find<Node>("Dst6"),
                            Names::Find<Node>("Dst7"),Names::Find<Node>("Dst8")};
  // 设置生产者为Dst3和Dst9。
  Ptr<Node> producerPor = Names::Find<Node>("Dst3");
  Ptr<Node> producerUnp = Names::Find<Node>("Dst9");


  for (int i = 0; i < 8; i++) {
    // 使得数据包的请求符合Zipf分布。
    ndn::AppHelper consumerHelper("ns3::ndn::ConsumerZipfMandelbrot");
    // 设置攻击频率位90,即每秒发送90个数据包。
    consumerHelper.SetAttribute("Frequency", StringValue("90")); 
    // 设置请求内容的前缀为“/popular”
    consumerHelper.SetPrefix("/popular");
    ApplicationContainer app = consumerHelper.Install(consumers[i]);
    app.Start(Seconds(0.01 * i));
  }

  for (int i = 10; i < 16; i++) {
      ndn::AppHelper consumerHelper("ns3::ndn::ConsumerZipfMandelbrot");
      consumerHelper.SetAttribute("Frequency", StringValue("80")); 

      consumerHelper.SetPrefix("/popular");
      ApplicationContainer app = consumerHelper.Install(consumers[i]);
      app.Start(Seconds(0.01 * i));
  }

  //=================================================================================================
  // ConsumerBatches节点能够实现在特定的时刻发送指定数量的数据包
  ndn::AppHelper consumerHelper("ns3::ndn::ConsumerBatches");
  // 30s 1000表示在第30秒发送1000个数据包
  consumerHelper.SetAttribute("Batches", StringValue("30s 1000 90s 1000 150s 1000 200s 1000 250s 1000 300s 800 400s 800 500s 800 600s 800 700s 800 800s 800 900s 800\
                                                      1000s 800 1500s 800 1600s 800 1700s 800 1800s 800 1900s 800 2000s 800 2150s 800 2200s 800 2400s 800 2500s 800\
                                                      2700s 800 2800s 800 2900s 800 2650s 800 1950s 100 1951s 100 1952s 100 1953s 100 1954s 100 1955s 100 1956s 100 1957s 100\
                                                      1958s 100 1959s 100 1960s 100 750s 100 751s 100 752s 100 753s 100 754s 100 755s 100 756s 100 757s 100 758s 100 759s 100 760s 100\
                                                      850s 100 851s 100 852s 100 853s 100 854s 100 855s 100 856s 100 857s 100 858s 100 859s 100 860s 100 3100s 800 3200s 800 3350s 800\
                                                      3400s 800 3500s 800 3600s 800 3650s 800 3700s 800 3800s 800 3900s 800 4000s 800 4100s 800 4200s 800 4300s 800\
                                                      4400s 800 4500s 800 4600s 800 4700s 800 4800s 800 4900s 800 5000s 800 5100s 800 5230s 800 5269s 800 5377s 800 5468s 800\
                                                      5555s 800 5684s 800 5778s 800 5850s 800 5900s 100 5901s 100 5902s 100 5903s 100 5904s 100 5905s 100 5906s 100 5907s 5\
                                                      5908s 100 5909s 100 6000s 100 6100s 800 6200s 800 6350s 800 6400s 800 6500s 800 6600s 800 6700s 750 6800s 900\
                                                      7000s 800 7100s 800 7200s 800 7301s 800 7400s 800 7500s 800 7600s 800 7700s 800 7800s 800 7890s 800 8810s 800\
                                                      8950s 800 8960s 800 9000s 800 9100s 800 9200s 800 9300s 800 9450s 800 9512s 800 9600s 800 9700s 800\
                                                      9800s 800 9897s 800 9950s 100 9951s 100 9952s 100 9953s 100 9954s 100 9955s 100 9956s 100 9957s 100"));
  // 设置攻击前缀为“/unpopular”
  consumerHelper.SetPrefix("/unpopular");
  ApplicationContainer app = consumerHelper.Install(consumers[8]);
  app.Start(Seconds(0.01 * 8));




  ndn::AppHelper consumerHelper2("ns3::ndn::ConsumerBatches");
  consumerHelper2.SetAttribute("Batches", StringValue("30s 1000 90s 1000 150s 1000 200s 1000 250s 1000 300s 800 400s 800 500s 800 600s 800 700s 800 800s 800 900s 800\
                                                      1000s 800 1500s 800 1600s 800 1700s 800 1800s 800 1900s 800 2000s 800 2150s 800 2200s 800 2400s 800 2500s 800\
                                                      2700s 800 2800s 800 2900s 800 2650s 800 1950s 100 1951s 100 1952s 100 1953s 100 1954s 100 1955s 100 1956s 100 1957s 100\
                                                      1958s 100 1959s 100 1960s 100 750s 100 751s 100 752s 100 753s 100 754s 100 755s 100 756s 100 757s 100 758s 100 759s 100 760s 100\
                                                      850s 100 851s 100 852s 100 853s 100 854s 100 855s 100 856s 100 857s 100 858s 100 859s 100 860s 100 3100s 800 3200s 800 3350s 800\
                                                      3400s 800 3500s 800 3600s 800 3650s 800 3700s 800 3800s 800 3900s 800 4000s 800 4100s 800 4200s 800 4300s 800\
                                                      4400s 800 4500s 800 4600s 800 4700s 800 4800s 800 4900s 800 5000s 800 5100s 800 5230s 800 5269s 800 5377s 800 5468s 800\
                                                      5555s 800 5684s 800 5778s 800 5850s 800 5900s 100 5901s 100 5902s 100 5903s 100 5904s 100 5905s 100 5906s 100 5907s 5\
                                                      5908s 100 5909s 100 6000s 100 6100s 800 6200s 800 6350s 800 6400s 800 6500s 800 6600s 800 6700s 750 6800s 900\
                                                      7000s 800 7100s 800 7200s 800 7301s 800 7400s 800 7500s 800 7600s 800 7700s 800 7800s 800 7890s 800 8810s 800\
                                                      8950s 800 8960s 800 9000s 800 9100s 800 9200s 800 9300s 800 9450s 800 9512s 800 9600s 800 9700s 800\
                                                      9800s 800 9897s 800 9950s 100 9951s 100 9952s 100 9953s 100 9954s 100 9955s 100 9956s 100 9957s 100"));
  consumerHelper2.SetPrefix("/unpopular");
  ApplicationContainer app2 = consumerHelper2.Install(consumers[9]);
  app2.Start(Seconds(0.01 * 9));

  //===================================================================================================

  ndn::AppHelper producerHelper("ns3::ndn::Producer");
  producerHelper.SetAttribute("PayloadSize", StringValue("1024"));


  ndnGlobalRoutingHelper.AddOrigins("/popular", producerPor);
  // 设置生产者生产的内容。
  producerHelper.SetPrefix("/popular");
  producerHelper.Install(producerPor);


  ndn::AppHelper producerHelper2("ns3::ndn::Producer");
  producerHelper2.SetAttribute("PayloadSize", StringValue("1024"));


  ndnGlobalRoutingHelper.AddOrigins("/unpopular", producerUnp);
  // 设置生产者生产的内容。
  producerHelper2.SetPrefix("/unpopular");
  producerHelper2.Install(producerUnp);

  //=================================================================================================

  // Calculate and install FIBs
  ndn::GlobalRoutingHelper::CalculateRoutes();

  // 模拟器在第10000秒停止，即运行10000秒。
  Simulator::Stop(Seconds(10000.0));
  // 将结果写入cs-trace.txt
  ndn::CsTracer::InstallAll("cs-trace.txt", Seconds(1));

  Simulator::Run();
  Simulator::Destroy();

  return 0;
}

} // namespace ns3

int
main(int argc, char* argv[])
{
  return ns3::main(argc, argv);
}
