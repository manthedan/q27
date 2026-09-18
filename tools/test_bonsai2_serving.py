#!/usr/bin/env python3
"""Bonsai's no-MTP HTTP smoke; unlike the q4s recovery gate, needs no chunking.

Usage: test_bonsai2_serving.py SERVER MODEL TOKENIZER
The generic recovery gate requires explicit snapshot hints, which upstream
intentionally ignores on serial-only Bonsai. Do not claim that gate passed.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"metal"))
from test_server_recovery import start_server, stop_server, request_json, request_sse


def require(condition, message):
    if not condition: raise RuntimeError(message)


def check_grammar_log(lines, label):
    engaged=[line for line in lines if "[toolgram] engaged" in line and "dialect=xml" in line]
    require(len(engaged)==1, f"{label}: expected one XML engagement, got {len(engaged)}")
    require(sum("[toolgram] call closed" in line for line in lines)==1,
            f"{label}: expected one grammar-closed call")
    require(not any("[toolgram] disengaged:" in line for line in lines),
            f"{label}: grammar unexpectedly dropped")


def main(server, model, tokenizer):
    with tempfile.TemporaryDirectory(prefix="q27-bonsai2-serving-") as cache:
        process,thread,lines,port=start_server(server,model,tokenizer,cache,{},context=2048,
                                              extra_args=("--constrain-tools",))
        try:
            require(any('xml (trained-format default)' in line for line in lines),
                    "Metal must initialize its per-model XML dialect before serving")
            status,models=request_json(port,"/v1/models")
            require(status==200 and models.get("data"),"models catalog")
            prompt=[{"role":"user","content":"Reply with exactly: ready"}]
            chat={"model":"bonsai2","messages":prompt,"max_tokens":32,"temperature":0}
            status,body=request_json(port,"/v1/chat/completions",chat)
            require(status==200 and body["choices"][0]["message"]["content"].strip()=="ready",f"Chat: {body}")
            status,body=request_json(port,"/v1/messages",chat)
            text="".join(x.get("text","") for x in body.get("content",[]) if x.get("type")=="text")
            require(status==200 and text.strip()=="ready",f"Messages: {body}")
            status,body=request_json(port,"/v1/messages/count_tokens",chat)
            require(status==200 and body.get("input_tokens",0)>0,"count_tokens")
            responses={"model":"bonsai2","input":"Reply with exactly: ready","max_output_tokens":32,"temperature":0}
            status,body=request_json(port,"/v1/responses",responses)
            text="".join(c.get("text","") for x in body.get("output",[]) for c in x.get("content",[]) if c.get("type")=="output_text")
            require(status==200 and text.strip()=="ready",f"Responses: {body}")
            status,headers,events=request_sse(port,"/v1/chat/completions",{**chat,"stream":True})
            text="".join(c.get("delta",{}).get("content","") for x in events for c in x.get("choices",[]))
            require(status==200 and len(headers)==1 and text.strip()=="ready","Chat SSE")
            bad={**chat,"max_tokens":1.5}
            status,_=request_json(port,"/v1/chat/completions",bad)
            require(status==400,"fractional token limit must reject")
            tool_request={**chat,"max_tokens":96,
                "messages":[{"role":"user","content":"Call echo with word marigold."}],
                "tools":[{"type":"function","function":{"name":"echo","description":"Echo a word",
                    "parameters":{"type":"object","properties":{"word":{"type":"string"}},"required":["word"]}}}],
                "tool_choice":{"type":"function","function":{"name":"echo"}}}
            status,body=request_json(port,"/v1/chat/completions",tool_request)
            require(status==200,f"tool request: {body}")
            calls=body["choices"][0]["message"].get("tool_calls",[])
            require(len(calls)==1 and calls[0]["function"]["name"]=="echo" and
                    json.loads(calls[0]["function"]["arguments"])=={"word":"marigold"},f"tool result: {body}")
            # FORCED deliberately bypasses the grammar. Exercise AUTO with
            # --constrain-tools through all six route/provider callsites too.
            def constrained(request, route, payload):
                # Requests are serial. Drain through this request's grammar
                # close before taking the next boundary; never credit a later
                # request with multiple engagements from an earlier route.
                start=len(lines)
                result=request(port,route,payload)
                deadline=time.monotonic()+5
                while not any("[toolgram] call closed" in line for line in lines[start:]) and time.monotonic()<deadline:
                    time.sleep(0.01)
                check_grammar_log(lines[start:], f"{route} stream={payload.get('stream',False)}")
                return result

            auto={**tool_request,"tool_choice":"auto"}
            status,body=constrained(request_json,"/v1/chat/completions",auto)
            calls=body["choices"][0]["message"].get("tool_calls",[])
            require(status==200 and len(calls)==1 and calls[0]["function"]["name"]=="echo" and
                    json.loads(calls[0]["function"]["arguments"])=={"word":"marigold"},f"constrained Chat: {body}")
            fn=auto["tools"][0]["function"]
            messages={**auto,"tool_choice":{"type":"auto"},
                "tools":[{"name":fn["name"],"description":fn["description"],"input_schema":fn["parameters"]}]}
            status,body=constrained(request_json,"/v1/messages",messages)
            calls=[x for x in body.get("content",[]) if x.get("type")=="tool_use"]
            require(status==200 and len(calls)==1 and calls[0]["name"]=="echo" and
                    calls[0]["input"]=={"word":"marigold"},f"constrained Messages: {body}")
            response_tools={"model":"bonsai2","input":"Call echo with word marigold.",
                "max_output_tokens":96,"temperature":0,
                "tools":[{"type":"function",**fn}],"tool_choice":"auto"}
            status,body=constrained(request_json,"/v1/responses",response_tools)
            calls=[x for x in body.get("output",[]) if x.get("type")=="function_call"]
            require(status==200 and len(calls)==1 and calls[0]["name"]=="echo" and
                    json.loads(calls[0]["arguments"])=={"word":"marigold"},f"constrained Responses: {body}")
            for route,payload in (("/v1/chat/completions",auto),("/v1/messages",messages),("/v1/responses",response_tools)):
                status,headers,events=constrained(request_sse,route,{**payload,"stream":True})
                require(status==200 and len(headers)==1 and events,f"constrained stream {route}")
                if route.endswith("completions"):
                    fragments=[call for event in events for choice in event.get("choices",[])
                               for call in choice.get("delta",{}).get("tool_calls",[])]
                    require(len({x.get("index") for x in fragments})==1 and
                            "".join(x.get("function",{}).get("name","") for x in fragments)=="echo",
                            f"one streamed echo call required: {events}")
                    args="".join(x.get("function",{}).get("arguments","") for x in fragments)
                elif route.endswith("messages"):
                    starts=[x["content_block"] for x in events if x.get("type")=="content_block_start" and
                            x.get("content_block",{}).get("type")=="tool_use"]
                    require(len(starts)==1 and starts[0]["name"]=="echo",f"one streamed echo call required: {events}")
                    args="".join(x.get("delta",{}).get("partial_json","") for x in events)
                else:
                    calls=[x["item"] for x in events if x.get("type")=="response.output_item.done" and
                           x.get("item",{}).get("type")=="function_call"]
                    require(len(calls)==1 and calls[0]["name"]=="echo",f"one streamed echo call required: {events}")
                    args="".join(x.get("delta","") for x in events if x.get("type")=="response.function_call_arguments.delta")
                    # The XML recovery path may publish a complete arguments.done
                    # rather than incremental deltas. Both are valid wire forms.
                    done=[x.get("arguments","") for x in events if x.get("type")=="response.function_call_arguments.done"]
                    require(len(done)==1,"one completed Responses function call required")
                    if not args: args=done[0]
                    else: require(args==done[0],"Responses deltas/done disagree")
                require(bool(args),f"missing streamed tool args {route}: {events}")
                require(json.loads(args)=={"word":"marigold"},f"streamed tool args {route}: {args}")
            empty={**auto,"messages":[{"role":"user","content":"Call ping with no arguments."}],
                "tools":[{"type":"function","function":{"name":"ping","description":"Ping",
                    "parameters":{"type":"object","properties":{},"additionalProperties":False}}}]}
            status,body=constrained(request_json,"/v1/chat/completions",empty)
            calls=body["choices"][0]["message"].get("tool_calls",[])
            require(status==200 and len(calls)==1 and calls[0]["function"]["name"]=="ping" and
                    json.loads(calls[0]["function"]["arguments"])=={},f"zero-argument schema: {body}")
            required_only={**auto,"tools":[{"type":"function","function":{**fn,
                "parameters":{"type":"object","required":["word"]}}}]}
            status,body=constrained(request_json,"/v1/chat/completions",required_only)
            calls=body["choices"][0]["message"].get("tool_calls",[])
            require(status==200 and len(calls)==1 and calls[0]["function"]["name"]=="echo" and
                    json.loads(calls[0]["function"]["arguments"])=={"word":"marigold"},f"required-only schema: {body}")
            open_schema={**auto,"tools":[{"type":"function","function":{**fn,
                "parameters":{"type":"object"}}}]}
            status,body=constrained(request_json,"/v1/chat/completions",open_schema)
            calls=body["choices"][0]["message"].get("tool_calls",[])
            require(status==200 and len(calls)==1 and calls[0]["function"]["name"]=="echo" and
                    json.loads(calls[0]["function"]["arguments"])=={"word":"marigold"},f"open object schema: {body}")
            print("Bonsai 2 APIs/SSE/count_tokens, six XML tool paths, closed-empty/open/required-only schemas: PASS")
        finally:
            stop_server(process,thread)
            if process.returncode not in (0,-15):sys.stderr.write("".join(lines))


if __name__=="__main__":
    if len(sys.argv)!=4:raise SystemExit(__doc__)
    main(*sys.argv[1:])
