#include "disk_snapshot_store.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

static SnapPeekInfo fixture_peek(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    uint8_t resident=0;
    uint32_t count=0;
    in.read(reinterpret_cast<char*>(&resident),sizeof resident);
    in.read(reinterpret_cast<char*>(&count),sizeof count);
    SnapPeekInfo info;
    info.logits_resident=resident!=0;
    info.position=count;
    info.tokens.resize(count);
    if(count) in.read(reinterpret_cast<char*>(info.tokens.data()),
                      (std::streamsize)count*sizeof(uint32_t));
    if(!in) throw std::runtime_error("invalid snapshot fixture");
    return info;
}

static void unused_hash(const uint32_t*,uint32_t,char out[41]) {
    for(int i=0;i<40;i++) out[i]='0';
    out[40]='\0';
}

static void write_file(const std::string& path,size_t bytes,int age_seconds) {
    std::ofstream out(path,std::ios::binary);
    std::string payload(bytes,'x');
    out.write(payload.data(),(std::streamsize)payload.size());
    out.close();
    std::error_code ec;
    fs::last_write_time(path,fs::file_time_type::clock::now()-
        std::chrono::seconds(age_seconds),ec);
}

static void publish_fixture(const std::string& path,
                            const std::vector<uint32_t>& tokens,bool resident) {
    const std::string tmp=path+".next";
    std::ofstream out(tmp,std::ios::binary);
    const uint8_t flag=resident?1:0;
    const uint32_t count=(uint32_t)tokens.size();
    out.write(reinterpret_cast<const char*>(&flag),sizeof flag);
    out.write(reinterpret_cast<const char*>(&count),sizeof count);
    if(count) out.write(reinterpret_cast<const char*>(tokens.data()),
                        (std::streamsize)count*sizeof(uint32_t));
    out.close();
    fs::rename(tmp,path);
}

int main() {
    const std::string dir=(::getenv("TMPDIR")?::getenv("TMPDIR"):"/tmp")+
        std::string("/q27-snapshot-isolation-")+std::to_string((long long)getpid());
    std::error_code ec;
    fs::remove_all(dir,ec);
    fs::create_directories(dir,ec);
    if(ec) {
        std::fprintf(stderr,"cannot create fixture directory: %s\n",ec.message().c_str());
        return 1;
    }

    const std::string own=dir+"/own-current.q27snap";
    const std::string foreign=dir+"/other-older.q27snap";
    write_file(own,100,10);
    write_file(foreign,200,100);

    DiskSnapshotStore store(&fixture_peek,&unused_hash);
    store.init(dir,100,"own-",false);
    const auto evicted=store.evict_past_budget();
    const bool eviction_ok=evicted.first==0 && fs::exists(own,ec) && fs::exists(foreign,ec);
    if(!eviction_ok) {
        std::fprintf(stderr,"snapshot eviction crossed artifact tag boundary\n");
        fs::remove_all(dir,ec);
        return 1;
    }

    fs::remove_all(dir,ec);
    fs::create_directories(dir,ec);
    DiskSnapshotStore shared(&fixture_peek,&unused_hash);
    shared.init(dir,0,"own-",false);
    const std::vector<uint32_t> tokens={1,2,3};
    const std::string path=shared.path_for(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,false);
    const bool saw_stale=!shared.exact_resident(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,true);
    const bool saw_external_resident=shared.exact_resident(tokens.data(),(uint32_t)tokens.size());
    publish_fixture(path,tokens,false);
    std::string found;
    uint32_t found_len=0;
    const bool rejected_external_stale=!shared.best_match(tokens,found,found_len);
    fs::remove_all(dir,ec);
    if(!saw_stale || !saw_external_resident || !rejected_external_stale) {
        std::fprintf(stderr,"snapshot metadata survived an external atomic replacement\n");
        return 1;
    }
    std::puts("snapshot store tag and metadata isolation: PASS");
    return 0;
}
