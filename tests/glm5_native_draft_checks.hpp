// Included by the real-GGUF echo fixture. These are local state/arithmetic
// checks, not an independent engine oracle, representative acceptance or TP.
// Norm/plain-residual equations follow the pinned SGLang NextN reference in
// the native-draft dossier. The first-row oracle assembles backend primitives
// independently; singleton attention needs no query/indexer selection.
static constexpr uint64_t native_row=4096u*4u, native_vocab=154880u*4u;

static void native_first_row_reference(ds4_glm5_next_exec_ctx &x,
        ds4_gpu_tensor *previous,unsigned token,ds4_gpu_tensor *hidden,ds4_gpu_tensor *logits) {
    const auto &l=x.model->layer[45]; const auto &m=l.mla; const auto &f=l.ffn_weight;
    const unsigned base=x.tp_rank*1024u, heads=32u, first=x.tp_rank*32u;
    Tensor embedding(native_row), concat(2*native_row), eh(native_row), norm(native_row);
    Tensor kv(512*4), kv_norm(512*4), query(heads*256*4), qk(heads*512*4);
    Tensor selected(4), values(heads*256*4), local(native_row), attention(native_row), residual(native_row);
    auto *en=ds4_gpu_tensor_view(concat,0,native_row);
    auto *hn=ds4_gpu_tensor_view(concat,native_row,native_row); REQUIRE(en && hn);
    REQUIRE(x.model->token_embd_type==30 && x.model->output_type==30);
    REQUIRE(ds4_gpu_embed_token_hc_bf16_tensor(embedding,x.model_map,x.model_size,
        x.model->token_embd,154880,token,4096,1));
    REQUIRE(ds4_gpu_rms_norm_weight_tensor(en,embedding,x.model_map,x.model_size,
        x.model->nextn_enorm,4096,x.model->rms_norm_eps));
    REQUIRE(ds4_gpu_rms_norm_weight_tensor(hn,previous,x.model_map,x.model_size,
        x.model->nextn_hnorm,4096,x.model->rms_norm_eps));
    REQUIRE(ds4_gpu_matmul_bf16_tensor(eh,x.model_map,x.model_size,
        x.model->nextn_eh_proj,8192,4096,concat,1));
    REQUIRE(ds4_gpu_rms_norm_weight_tensor(norm,eh,x.model_map,x.model_size,
        l.attn_norm,4096,x.model->rms_norm_eps));
    REQUIRE(ds4_gpu_matmul_q8_0_tensor(kv,x.model_map,x.model_size,m.kv_a_mqa,4096,512,norm,1));
    REQUIRE(ds4_gpu_glm_kv_lora_rms_norm_tensor(kv_norm,kv,x.model_map,x.model_size,
        m.kv_a_norm,1,512,512,x.model->rms_norm_eps));
    REQUIRE(ds4_gpu_tensor_fill_f32(query,0,heads*256) &&
        ds4_gpu_tensor_fill_f32(qk,0,heads*512) &&
        ds4_gpu_glm_fill_selected_range_tensor(selected,1));
    const uint64_t vb=m.v_b+(uint64_t)first*256*(512/32)*34;
    REQUIRE(ds4_gpu_glm_attention_indexed_decode_typed_tensor(values,query,qk,kv_norm,nullptr,
        x.model_map,x.model_size,vb,8,selected,1,1,false,heads,512,256,0,256,0,
        1,1,0,1,0,0));
    REQUIRE(ds4_gpu_matmul_q8_0_kslice_tensor(local,x.model_map,x.model_size,m.output,
        64*256,(uint64_t)first*256,32*256,4096,values,0));
    REQUIRE(ds4_gpu_add_tensor(attention,local,local,4096) &&
        ds4_gpu_add_tensor(residual,eh,attention,4096));
    REQUIRE(ds4_gpu_rms_norm_weight_tensor(norm,residual,x.model_map,x.model_size,
        l.ffn_norm,4096,x.model->rms_norm_eps));
    Tensor router(288*4), probs(288*4), experts(8*4), weights(8*4);
    REQUIRE(ds4_gpu_matmul_f32_tensor(router,x.model_map,x.model_size,f.gate_inp,4096,288,norm,1));
    REQUIRE(ds4_gpu_glm_router_select_tensor(experts,weights,probs,x.model_map,x.model_size,
        f.exp_probs_b,router,288,8,2.5f));
    Tensor gate(8*1024*4), up(8*1024*4), mid(8*1024*4), routed_rows(8*native_row), routed(native_row);
    ds4_gpu_q4k_window_cache_config config={};
    config.model_map=x.model_map; config.gate_offset=f.gate_exps;
    config.up_offset=f.up_exps; config.down_offset=f.down_exps; config.n_expert=288;
    config.gate_row_base=base; config.gate_row_count=1024; config.gate_column_byte_count=2304;
    config.down_row_count=4096; config.down_column_byte_base=x.tp_rank*576;
    config.down_column_byte_count=576; config.slots=8;
    auto *window=ds4_gpu_q4k_window_cache_create(&config); REQUIRE(window);
    REQUIRE(ds4_gpu_routed_moe_one_packed_q4k_window_tensor(routed,gate,up,mid,routed_rows,
        window,experts,weights,8,10,norm,nullptr,45));
    Tensor sg(1024*4), su(1024*4), sm(1024*4), shared(native_row), sum(native_row), both(native_row);
    const char *pair=std::getenv("DS4_ROCM_GLM5_SHARED_Q8_PAIR_DECODE");
    if(pair && !std::strcmp(pair,"1")) {
        REQUIRE(ds4_gpu_matmul_q8_0_pair_tensor(sg,su,x.model_map,x.model_size,
            f.gate_shexp+(uint64_t)base*4352,f.up_shexp+(uint64_t)base*4352,4096,1024,1024,norm,1));
    } else {
        REQUIRE(ds4_gpu_matmul_q8_0_tensor(sg,x.model_map,x.model_size,
            f.gate_shexp+(uint64_t)base*4352,4096,1024,norm,1));
        REQUIRE(ds4_gpu_matmul_q8_0_tensor(su,x.model_map,x.model_size,
            f.up_shexp+(uint64_t)base*4352,4096,1024,norm,1));
    }
    REQUIRE(ds4_gpu_swiglu_tensor(sm,sg,su,1024,10,1));
    REQUIRE(ds4_gpu_matmul_q8_0_kslice_tensor(shared,x.model_map,x.model_size,f.down_shexp,
        2048,base,1024,4096,sm,0));
    REQUIRE(ds4_gpu_add_tensor(sum,routed,shared,4096) &&
        ds4_gpu_add_tensor(both,sum,sum,4096) && ds4_gpu_add_tensor(eh,residual,both,4096));
    REQUIRE(ds4_gpu_rms_norm_weight_tensor(hidden,eh,x.model_map,x.model_size,
        x.model->nextn_shared_head_norm,4096,x.model->rms_norm_eps));
    REQUIRE(ds4_gpu_matmul_bf16_tensor(logits,x.model_map,x.model_size,x.model->output,4096,154880,hidden,1));
    REQUIRE(ds4_gpu_synchronize());
    ds4_gpu_q4k_window_cache_destroy(window);
    ds4_gpu_tensor_free(hn); ds4_gpu_tensor_free(en);
}

static void native_draft_case(ds4_glm5_next_exec_ctx &x,unsigned m,unsigned prefix,unsigned accepted) {
    State serial(*x.model,true), candidate(*x.model,true);
    Workspace sw(1,true), cw(1,true);
    seed_mla(serial,45,prefix); seed_mla(candidate,45,prefix);
    Tensor input(native_row), a(native_row), b(native_row), la(native_vocab), lb(native_vocab);
    std::vector<float> start(4096);
    for(unsigned i=0;i<4096;++i) start[i]=float(int((i*193+prefix*17)%997)-498)/501.3f;
    REQUIRE(ds4_gpu_tensor_write(input,0,start.data(),native_row));
    std::vector<std::vector<float>> hidden, logits;
    std::vector<unsigned> tokens(m+1); tokens[0]=73;
    for(unsigned i=0;i<m;++i) {
        REQUIRE(ds4_glm5_next_draft_step(&x,&serial.s.mla[45],sw,input,tokens[i],a,la));
        hidden.push_back(read(a,4096)); logits.push_back(read(la,154880));
        tokens[i+1]=(unsigned)(std::max_element(logits.back().begin(),logits.back().end())-logits.back().begin());
        REQUIRE(ds4_gpu_tensor_copy(input,0,a,0,native_row));
    }
    REQUIRE(ds4_glm5_next_mla_replay_reserve(&candidate.s.mla[45],m));
    ds4_glm5_next_mla_state *view=nullptr;
    REQUIRE(ds4_glm5_next_mla_verify_begin(&candidate.s.mla[45],m,&view));
    REQUIRE(ds4_gpu_tensor_write(input,0,start.data(),native_row));
    for(unsigned i=0;i<m;++i) {
        REQUIRE(ds4_glm5_next_draft_step(&x,view,cw,input,tokens[i],b,lb));
        const auto bh=read(b,4096), bl=read(lb,154880);
        REQUIRE(!std::memcmp(bh.data(),hidden[i].data(),native_row) &&
            !std::memcmp(bl.data(),logits[i].data(),native_vocab));
        compared_values+=4096+154880;
        REQUIRE(ds4_gpu_tensor_copy(input,0,b,0,native_row));
    }
    REQUIRE(candidate.s.mla[45].token_count==prefix &&
        ds4_glm5_next_mla_verify_finish(&candidate.s.mla[45],accepted));
    REQUIRE(ds4_glm5_next_state_reset(&serial.s)); seed_mla(serial,45,prefix);
    REQUIRE(ds4_gpu_tensor_write(input,0,start.data(),native_row));
    for(unsigned i=0;i<accepted;++i) {
        REQUIRE(ds4_glm5_next_draft_step(&x,&serial.s.mla[45],sw,input,tokens[i],a,la));
        REQUIRE(ds4_gpu_tensor_copy(input,0,a,0,native_row));
    }
    equal_layer(serial,candidate,45);
    // A correction may differ from the proposal. Both caches must produce
    // identical following hidden/logits from the accepted predecessor hidden.
    const unsigned correction=(tokens[accepted]+113)%154880;
    REQUIRE(ds4_glm5_next_draft_step(&x,&serial.s.mla[45],sw,input,correction,a,la));
    REQUIRE(ds4_glm5_next_draft_step(&x,&candidate.s.mla[45],cw,input,correction,b,lb));
    equal("draft-continuation-hidden",a,b,4096); equal("draft-continuation-logits",la,lb,154880);
    equal_layer(serial,candidate,45); ++cases;
    std::printf("NATIVE_CASE rank=%u M=%u prefix=%u accepted=%u bitwise=pass\n",x.tp_rank,m,prefix,accepted);
    std::fflush(stdout);
}

static void native_draft_checks(ds4_glm5_next_exec_ctx &x) {
    REQUIRE(std::getenv("DS4_GLM5_MLA_OWNED_HEADS") &&
        !std::strcmp(std::getenv("DS4_GLM5_MLA_OWNED_HEADS"),"1"));
    State s(*x.model,true); Workspace w(1,true);
    Tensor hc(hc_row), previous(native_row), h(native_row), rh(native_row), logits(native_vocab), rl(native_vocab);
    Tensor mean(4*4), contracted(native_row);
    REQUIRE(ds4_gpu_tensor_fill_f32(mean,0.25f,4));
    for(unsigned mode=0;mode<3;++mode) {
        REQUIRE(ds4_glm5_next_state_reset(&s.s));
        std::vector<float> input(16384);
        for(unsigned i=0;i<input.size();++i)
            input[i]=mode==0 ? 0.0f : float(int((i*193+mode*71)%997)-498)/(mode==1?1001.3f:11.3f);
        REQUIRE(ds4_gpu_tensor_write(hc,0,input.data(),hc_row));
        REQUIRE(ds4_glm5_next_draft_target_hidden(&x,w,hc,previous));
        REQUIRE(ds4_gpu_hc_weighted_sum_tensor(contracted,hc,mean,4096,4) &&
            ds4_gpu_rms_norm_weight_tensor(rh,contracted,x.model_map,x.model_size,
                x.model->output_norm,4096,x.model->rms_norm_eps));
        equal("target-hidden-contract",previous,rh,4096);
        REQUIRE(ds4_glm5_next_draft_step(&x,&s.s.mla[45],w,previous,73+mode,h,logits));
        native_first_row_reference(x,previous,73+mode,rh,rl);
        equal("native-plain-reference-hidden",h,rh,4096);
        equal("native-plain-reference-logits",logits,rl,154880);
    }
    // Reject binding changes and aliases without consuming a state row or gate.
    const unsigned frontier=s.s.mla[45].token_count, calls=x.tp->calls;
    auto clone=s.s.mla[45];
    REQUIRE(!ds4_glm5_next_draft_step(&x,&clone,w,previous,73,h,logits));
    REQUIRE(!ds4_glm5_next_draft_step(&x,&s.s.mla[45],w,previous,73,previous,logits));
    auto other=x; other.model_size--;
    REQUIRE(!ds4_glm5_next_draft_step(&other,&s.s.mla[45],w,previous,73,h,logits));
    other=x; uint64_t alternate_sequence=*x.tp_sequence; other.tp_sequence=&alternate_sequence;
    REQUIRE(!ds4_glm5_next_draft_step(&other,&s.s.mla[45],w,previous,73,h,logits));
    State another(*x.model,true);
    REQUIRE(!ds4_glm5_next_draft_step(&x,&another.s.mla[45],w,previous,73,h,logits));
    REQUIRE(s.s.mla[45].token_count==frontier && x.tp->calls==calls && s.s.valid);
    // Each payload exchange can fail; both must invalidate the private owner.
    for(unsigned fail=1;fail<=2;++fail) {
        REQUIRE(ds4_glm5_next_state_reset(&s.s)); x.tp->fail_call=x.tp->calls+fail;
        REQUIRE(!ds4_glm5_next_draft_step(&x,&s.s.mla[45],w,previous,73,h,logits) && !s.s.valid);
        x.tp->fail_call=0; x.tp->failed=false;
    }
    REQUIRE(ds4_glm5_next_state_reset(&s.s));
    for(unsigned m : {2u,4u,8u}) for(unsigned accepted=0;accepted<=m;++accepted)
        native_draft_case(x,m,3,accepted);
    for(unsigned prefix : {0u,2047u,2048u,8192u,8193u}) for(unsigned accepted=0;accepted<=4;++accepted)
        native_draft_case(x,4,prefix,accepted);
    REQUIRE(x.tp->aux_calls==0);
    // Local draft-only cost, including full head and host greedy selection.
    // Synthetic populated cache; excludes prompt warming, target and network.
    for(unsigned prefix : {0u,8192u}) {
        std::vector<double> samples;
        for(unsigned iteration=0;iteration<13;++iteration) {
            REQUIRE(ds4_glm5_next_state_reset(&s.s)); seed_mla(s,45,prefix);
            REQUIRE(ds4_gpu_tensor_fill_f32(previous,0.0625f,4096) && ds4_gpu_synchronize());
            unsigned token=73;
            const auto begin=std::chrono::steady_clock::now();
            for(unsigned step=0;step<8;++step) {
                REQUIRE(ds4_glm5_next_draft_step(&x,&s.s.mla[45],w,previous,token,h,logits));
                const auto host_logits=read(logits,154880);
                token=(unsigned)(std::max_element(host_logits.begin(),host_logits.end())-host_logits.begin());
                REQUIRE(ds4_gpu_tensor_copy(previous,0,h,0,native_row));
            }
            REQUIRE(ds4_gpu_synchronize());
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/8.0;
            if(iteration>=4) {
                samples.push_back(ms);
                std::printf("NATIVE_SAMPLE rank=%u prefix=%u sample=%u ms_per_draft=%.6f\n",x.tp_rank,prefix,iteration-4,ms);
            }
        }
        std::sort(samples.begin(),samples.end());
        std::printf("NATIVE_MEDIAN rank=%u prefix=%u ms_per_draft=%.6f model_tps=unmeasured\n",x.tp_rank,prefix,samples[4]);
        std::fflush(stdout);
    }
    std::puts("PASS native draft equations, chained logits, accepted-prefix state and continuation; simulated_peer=echo network_test=0 quality_test=0");
}
