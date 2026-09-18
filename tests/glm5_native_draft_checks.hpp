// Included by the real-GGUF echo fixture. These are local state/arithmetic
// checks, not an independent engine oracle, representative acceptance or TP.
// Norm/plain-residual equations follow the pinned SGLang NextN reference in
// the native-draft dossier. The first-row oracle assembles backend primitives
// independently; singleton attention needs no query/indexer selection.
static constexpr uint64_t native_row=4096u*4u, native_vocab=154880u*4u;

static void warm_difference(const char *name,ds4_gpu_tensor *a,ds4_gpu_tensor *b,uint64_t count) {
    auto av=read(a,count),bv=read(b,count);
    double square=0,ref_square=0,max_abs=0; uint64_t different=0;
    for(size_t i=0;i<av.size();++i) {
        const double d=(double)bv[i]-av[i];
        max_abs=std::max(max_abs,std::fabs(d)); square+=d*d; ref_square+=(double)av[i]*av[i];
        different+=std::memcmp(&av[i],&bv[i],4)!=0;
    }
    std::printf("WARM_DRIFT tensor=%s count=%llu differing=%llu max_abs=%.9g rel_l2=%.9g\n",
        name,(unsigned long long)count,(unsigned long long)different,max_abs,
        std::sqrt(square/std::max(ref_square,1e-30)));
}

static void warm_compare_metadata(State &a,State &b) {
    auto &x=a.s.mla[45], &y=b.s.mla[45];
    REQUIRE(x.token_count==y.token_count && x.complete_pools==y.complete_pools &&
        x.tail_count==y.tail_count && a.s.valid && b.s.valid);
    for(auto item : {std::make_pair(x.index_pool_ids,y.index_pool_ids),
                    std::make_pair(x.index_pool_valid,y.index_pool_valid)}) {
        const uint64_t bytes=(item.first==x.index_pool_ids?16u:4u)*x.complete_pools;
        std::vector<unsigned char> aa(bytes),bb(bytes);
        if(bytes) REQUIRE(ds4_gpu_tensor_read(item.first,0,aa.data(),bytes) &&
            ds4_gpu_tensor_read(item.second,0,bb.data(),bytes));
        REQUIRE(aa==bb);
    }
}

static void warm_case(ds4_glm5_next_exec_ctx &x,unsigned n,unsigned prefix,bool batch) {
    State ref(*x.model,true), got(*x.model,true);
    Workspace sw(1,true), ww(batch?n:1,true), hw(n,true), gw(1,true);
    seed_mla(ref,45,prefix); seed_mla(got,45,prefix);
    Tensor hc((uint64_t)n*hc_row), input((uint64_t)n*native_row), scalar_hidden(native_row);
    Tensor h(native_row), gh(native_row), logits(native_vocab), gl(native_vocab);
    std::vector<float> host((size_t)n*16384u); std::vector<uint32_t> tokens(n);
    for(unsigned t=0;t<n;++t) {
        tokens[t]=73+(t*37+prefix)%154000;
        for(unsigned j=0;j<16384;++j)
            host[(size_t)t*16384+j]=float(int((j*193+t*317+prefix*17)%997)-498)/501.3f;
    }
    REQUIRE(ds4_gpu_tensor_write(hc,0,host.data(),host.size()*4));
    REQUIRE(ds4_glm5_next_draft_target_hidden(&x,hw,hc,input));
    for(unsigned t=0;t<n;++t) {
        auto *hc_view=ds4_gpu_tensor_view(hc,(uint64_t)t*hc_row,hc_row);
        auto *row=ds4_gpu_tensor_view(input,(uint64_t)t*native_row,native_row);
        REQUIRE(hc_view && row && ds4_glm5_next_draft_target_hidden(&x,sw,hc_view,scalar_hidden));
        equal("warm-target-normalization",row,scalar_hidden,4096);
        if(batch) REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&ref.s,sw,row,&tokens[t],1));
        else REQUIRE(ds4_glm5_next_draft_step(&x,&ref.s.mla[45],sw,row,tokens[t],h,logits));
        ds4_gpu_tensor_free(row); ds4_gpu_tensor_free(hc_view);
    }
    const unsigned gates=x.tp->calls; const uint64_t seq=*x.tp_sequence;
    if(batch) REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&got.s,ww,input,tokens.data(),n));
    else for(unsigned t=0;t<n;++t) {
        auto *row=ds4_gpu_tensor_view(input,(uint64_t)t*native_row,native_row); REQUIRE(row);
        REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&got.s,ww,row,&tokens[t],1));
        ds4_gpu_tensor_free(row);
    }
    REQUIRE(x.tp->calls==gates && *x.tp_sequence==seq);
    warm_compare_metadata(ref,got);
    std::printf("WARM_CASE rank=%u M=%u prefix=%u mode=%s\n",x.tp_rank,n,prefix,
        batch?"batch-diagnostic":"scalar-exact");
    if(batch) {
        warm_difference("kv",ref.s.mla[45].compact_kv,got.s.mla[45].compact_kv,(uint64_t)(prefix+n)*512);
        warm_difference("pool",ref.s.mla[45].index_pool,got.s.mla[45].index_pool,ref.s.mla[45].complete_pools*128);
        warm_difference("key-tail",ref.s.mla[45].index_tail,got.s.mla[45].index_tail,4*128);
        warm_difference("gate-tail",ref.s.mla[45].pool_gate_tail,got.s.mla[45].pool_gate_tail,4*128);
    } else equal_layer(ref,got,45);
    REQUIRE(ds4_glm5_next_draft_step(&x,&ref.s.mla[45],sw,scalar_hidden,113,h,logits));
    REQUIRE(ds4_glm5_next_draft_step(&x,&got.s.mla[45],gw,scalar_hidden,113,gh,gl));
    if(batch) {
        warm_difference("next-hidden",h,gh,4096); warm_difference("next-logits",logits,gl,154880);
    } else {
        equal("warm-next-hidden",h,gh,4096); equal("warm-next-logits",logits,gl,154880);
        equal_layer(ref,got,45);
    }
    ++cases; std::fflush(stdout);
}

static void native_warm_checks(ds4_glm5_next_exec_ctx &x) {
    for(unsigned prefix : {0u,1u,2u,3u,2047u,2048u,8191u,8192u}) warm_case(x,13,prefix,false);
    for(unsigned n : {2u,4u,8u,17u,255u,256u}) for(unsigned prefix : {0u,3u,2047u,8192u})
        warm_case(x,n,prefix,true);
    // Pool batch publication must retain physical tails, including at aligned
    // boundaries. Changing this selector does not change projection arithmetic.
    setenv("DS4_ROCM_GLM5_BATCH_POOL_STAGE","1",1);
    warm_case(x,4,0,true); warm_case(x,256,8192,true);
    unsetenv("DS4_ROCM_GLM5_BATCH_POOL_STAGE");
    State s(*x.model,true); Workspace sw(1,true), bw(256,true);
    Tensor input(256*native_row); REQUIRE(ds4_gpu_tensor_fill_f32(input,0.0625f,256*4096));
    std::vector<uint32_t> tokens(256,73);
    auto *one=ds4_gpu_tensor_view(input,0,native_row); REQUIRE(one);
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&x,&s.s,bw,input,tokens.data(),255));
    tokens[255]=154880;
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&x,&s.s,bw,input,tokens.data(),256) && !s.s.mla[45].token_count);
    tokens[255]=73;
    REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&s.s,sw,one,tokens.data(),1));
    auto other=x; other.model_size--;
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&other,&s.s,sw,one,tokens.data(),1));
    REQUIRE(ds4_glm5_next_mla_replay_reserve(&s.s.mla[45],4));
    ds4_glm5_next_mla_state *view=nullptr;
    REQUIRE(ds4_glm5_next_mla_verify_begin(&s.s.mla[45],4,&view));
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&x,&s.s,sw,one,tokens.data(),1));
    REQUIRE(ds4_glm5_next_state_reset(&s.s)); seed_mla(s,45,context-255);
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&x,&s.s,bw,input,tokens.data(),256));
    // Inject a backend range refusal after KV has been written. Only the
    // copied test metadata changes; the original model bytes stay untouched.
    auto broken_model=*x.model; broken_model.layer[45].mla.index_k=x.model_size-1;
    auto broken_ctx=x; broken_ctx.model=&broken_model;
    State broken(broken_model,true); Workspace broken_w(1,true);
    REQUIRE(!ds4_glm5_next_draft_warm_rows(&broken_ctx,&broken.s,broken_w,one,tokens.data(),1) &&
        !broken.s.valid && !broken.s.mla[45].valid);
    ds4_gpu_tensor_free(one);
    // Four warmups + nine alternating scalar/batch samples. All hidden and
    // token preparation, reset and cache seeding are outside the timed region.
    State scalar(*x.model,true), batched(*x.model,true);
    Workspace scalar_w(1,true), batch_w(256,true);
    std::vector<double> samples[2];
    for(unsigned iteration=0;iteration<13;++iteration) for(unsigned arm=0;arm<2;++arm) {
        const unsigned mode=(arm+iteration)%2;
        State &state=mode?batched:scalar; Workspace &workspace=mode?batch_w:scalar_w;
        REQUIRE(ds4_glm5_next_state_reset(&state.s)); seed_mla(state,45,8192);
        REQUIRE(ds4_gpu_synchronize()); const auto begin=std::chrono::steady_clock::now();
        if(mode) REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&state.s,workspace,input,tokens.data(),256));
        else for(unsigned t=0;t<256;++t) {
            auto *row=ds4_gpu_tensor_view(input,(uint64_t)t*native_row,native_row); REQUIRE(row);
            REQUIRE(ds4_glm5_next_draft_warm_rows(&x,&state.s,workspace,row,&tokens[t],1));
            ds4_gpu_tensor_free(row);
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(iteration>=4) {
            samples[mode].push_back(ms);
            std::printf("WARM_SAMPLE rank=%u M=256 sample=%u mode=%u ms=%.6f\n",x.tp_rank,iteration-4,mode,ms);
        }
    }
    for(unsigned mode=0;mode<2;++mode) {
        std::sort(samples[mode].begin(),samples[mode].end());
        std::printf("WARM_MEDIAN rank=%u M=256 mode=%u ms=%.6f model_prefill_tps=unmeasured\n",
            x.tp_rank,mode,samples[mode][4]);
    }
    std::puts("PASS scalar warm equivalence and batch finite diagnostic; simulated_peer=echo quality_test=0 network_test=0");
}

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
