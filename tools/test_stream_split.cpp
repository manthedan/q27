// Streaming splitter regression (issue #4): a stray </tool_call> in the TEXT
// channel (bare-call wrapper leftover) must be stripped, not leaked, while real
// <tool_call>...</tool_call> pairs, split tags, and <think> still route right.
// Build: g++ -std=c++17 -I src tools/test_stream_split.cpp -o build/test_stream_split
#include "stream_split.h"
#include <cstdio>
using namespace q27;
static std::string collect(StreamSplitter& s, std::initializer_list<std::string> pieces, StreamSplitter::Chan want){
  std::string r;
  for(auto&p:pieces) for(auto&[ch,t]:s.feed(p)) if(ch==want) r+=t;
  for(auto&[ch,t]:s.flush()) if(ch==want) r+=t;
  return r;
}
static void check(const char*name,const std::string&got,const std::string&want){
  printf("  %-42s %s\n",name, got==want?"PASS":("FAIL got="+got+" want="+want).c_str());
}
int main(){
  { StreamSplitter s; auto t=collect(s,{"hello","</tool_call>","world"},StreamSplitter::TEXT); check("stray close stripped",t,"helloworld"); }
  { StreamSplitter s; auto t=collect(s,{"{\"name\": \"read\"}\n</tool_call>\n{\"name\": \"read2\"}\n</tool_call>\n"},StreamSplitter::TEXT);
    check("faisal multi: no </tool_call> in text", t.find("</tool_call>")==std::string::npos?"ok":"leak","ok"); }
  { StreamSplitter s; auto tool=collect(s,{"<tool_call>","{\"a\":1}","</tool_call>"},StreamSplitter::TOOL); check("normal pair -> TOOL",tool,"{\"a\":1}"); }
  { StreamSplitter s; auto tx=collect(s,{"<tool_call>","{\"a\":1}","</tool_call>","tail"},StreamSplitter::TEXT); check("normal pair TEXT=tail only",tx,"tail"); }
  { StreamSplitter s; auto t=collect(s,{"abc</tool","_call>def"},StreamSplitter::TEXT); check("split stray tag held+stripped",t,"abcdef"); }
  { StreamSplitter s; auto th=collect(s,{"<think>","reason","</think>","ans"},StreamSplitter::THINK); check("think still routes",th,"reason"); }
  { StreamSplitter s; auto tx=collect(s,{"<think>","reason","</think>","ans"},StreamSplitter::TEXT); check("think text=ans",tx,"ans"); }
  // enable_thinking: the <think>\n opener is prompt-injected (not generated), so the
  // generation paths start the splitter already in THINK. The model's reasoning
  // routes to THINK and its </think> flips to TEXT for the answer -- no opening tag
  // ever appears in the generated stream.
  { StreamSplitter s; s.chan=StreamSplitter::THINK; auto th=collect(s,{"reason","ing","</think>","\n\nans"},StreamSplitter::THINK); check("preseeded THINK: reasoning routes",th,"reasoning"); }
  { StreamSplitter s; s.chan=StreamSplitter::THINK; auto tx=collect(s,{"reason","ing","</think>","\n\nans"},StreamSplitter::TEXT); check("preseeded THINK: answer after </think>",tx,"\n\nans"); }
  { StreamSplitter s; s.native_xml=true; auto tool=collect(s,{"<func","tion=read><parameter=path>x</parameter>","</function>"},StreamSplitter::TOOL); check("native XML pair -> TOOL",tool,"<function=read><parameter=path>x</parameter></function>"); }
  { StreamSplitter s; s.native_xml=true; auto tx=collect(s,{"<function=read><parameter=path>x</parameter></function>"},StreamSplitter::TEXT); check("native XML emits no visible TEXT",tx,""); }
  { StreamSplitter s; s.native_xml=true; auto tool=collect(s,{"   <function=read><parameter=path>x</parameter></function>"},StreamSplitter::TOOL); check("three-space native call allowed",tool,"<function=read><parameter=path>x</parameter></function>"); }
  { StreamSplitter s; s.native_xml=true; auto tx=collect(s,{"    <function=read><parameter=path>x</parameter></function>"},StreamSplitter::TEXT); check("indented native example stays TEXT",tx,"    <function=read><parameter=path>x</parameter></function>"); }
  { StreamSplitter s; s.native_xml=true; auto tx=collect(s,{"Quoted:\n\"\n","<function=read><parameter=path>x</parameter></function>\n\""},StreamSplitter::TEXT); check("quoted native example stays TEXT",tx,"Quoted:\n\"\n<function=read><parameter=path>x</parameter></function>\n\""); }
  { StreamSplitter s; s.native_xml=true; s.chan=StreamSplitter::THINK; auto tool=collect(s,{"reason","</think>\n","<function=read><parameter=path>x</parameter></function>"},StreamSplitter::TOOL); check("native call after reasoning -> TOOL",tool,"<function=read><parameter=path>x</parameter></function>"); }
  { StreamSplitter s; s.native_xml=true; auto out=s.feed("<function=read></function><function=write></function>");
    int tools=0,bounds=0; std::string joined;
    for(auto&[ch,t]:out) { if(ch==StreamSplitter::TOOL){ tools++; joined+=t; } else if(ch==StreamSplitter::TEXT&&t.empty()) bounds++; }
    check("adjacent native calls stay separated",tools==2&&bounds==1&&joined=="<function=read></function><function=write></function>"?"ok":"bad","ok"); }
}
