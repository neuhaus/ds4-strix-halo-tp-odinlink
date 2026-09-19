// Default-off GLM generic projection experiment. Plans contain descriptors,
// never weights. Only the measured gfx1151 shapes and solution are admitted.
struct glm5_bf16_lt_plan {
    uint32_t k = 0, m = 0;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo[2] = {};
};
static std::vector<glm5_bf16_lt_plan> g_glm5_bf16_lt_plans;
static uint16_t *g_glm5_bf16_lt_activations;
static constexpr size_t glm5_bf16_lt_scratch_bytes = 32u * 1024u * 1024u;

static void glm5_bf16_lt_destroy_plan(glm5_bf16_lt_plan &p) {
    if (p.c) (void)hipblasLtMatrixLayoutDestroy(p.c);
    if (p.b) (void)hipblasLtMatrixLayoutDestroy(p.b);
    if (p.a) (void)hipblasLtMatrixLayoutDestroy(p.a);
    if (p.desc) (void)hipblasLtMatmulDescDestroy(p.desc);
    p = {};
}

static void glm5_bf16_lt_cleanup(void) {
    for (auto &p : g_glm5_bf16_lt_plans) glm5_bf16_lt_destroy_plan(p);
    g_glm5_bf16_lt_plans.clear();
    if (g_glm5_bf16_lt_activations) {
        int saved = -1;
        (void)hipGetDevice(&saved);
        if (hipSetDevice(0) == hipSuccess)
            (void)hipFree(g_glm5_bf16_lt_activations);
        g_glm5_bf16_lt_activations = nullptr;
        if (saved >= 0) (void)hipSetDevice(saved);
    }
}

static glm5_bf16_lt_plan *glm5_bf16_lt_get_plan(uint32_t k, uint32_t m) {
    if (!g_hipblaslt_ready || (k != 4096u && k != 8192u) ||
        (m != 256u && m != 1024u)) return nullptr;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != 0) return nullptr;
    for (auto &p : g_glm5_bf16_lt_plans)
        if (p.k == k && p.m == m) return &p;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) != hipSuccess ||
        strncmp(prop.gcnArchName,"gfx1151",7) != 0) return nullptr;

    glm5_bf16_lt_plan p;
    p.k = k;
    p.m = m;
    bool ready = false;
    do {
        if (!hipblaslt_ok(hipblasLtMatmulDescCreate(&p.desc,HIPBLAS_COMPUTE_32F,HIP_R_32F),"GLM BF16 desc")) break;
        const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        if (!hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"GLM BF16 transA") ||
            !hipblaslt_ok(hipblasLtMatmulDescSetAttribute(p.desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"GLM BF16 transB") ||
            !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.a,HIP_R_16BF,k,4096u,k),"GLM BF16 A") ||
            !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.b,HIP_R_16BF,k,m,k),"GLM BF16 B") ||
            !hipblaslt_ok(hipblasLtMatrixLayoutCreate(&p.c,HIP_R_32F,4096u,m,4096u),"GLM BF16 C")) break;
        // This index is tied to the frozen research toolchain. Also check the
        // solution family and query each beta case; never silently substitute.
        std::vector<int> indices{1176};
        std::vector<hipblasLtMatmulHeuristicResult_t> choices;
        if (!hipblaslt_ok(hipblaslt_ext::getAlgosFromIndex(g_hipblaslt,indices,choices),"GLM BF16 solution") ||
            choices.size() != 1u || choices[0].state != HIPBLAS_STATUS_SUCCESS) break;
        auto algorithm = choices[0].algo;
        const std::string name = hipblaslt_ext::getSolutionNameFromAlgo(g_hipblaslt,algorithm);
        const char *prefix = "Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT32x96x32_MI16x16x1_";
        if (hipblaslt_ext::getIndexFromAlgo(algorithm) != 1176 ||
            name.compare(0,strlen(prefix),prefix) != 0 ||
            name.find("_ISA1151_") == std::string::npos) break;
        const float alpha = 1.0f;
        bool supported = true;
        for (unsigned pass = 0; pass < 2; ++pass) {
            const float beta = float(pass);
            size_t workspace = SIZE_MAX;
            p.algo[pass] = algorithm;
            if (!hipblaslt_ok(hipblaslt_ext::matmulIsAlgoSupported(g_hipblaslt,p.desc,&alpha,
                    p.a,p.b,&beta,p.c,p.c,p.algo[pass],workspace),"GLM BF16 support") || workspace != 0u) {
                supported = false;
                break;
            }
        }
        if (!supported) break;
        fprintf(stderr,"ds4: ROCm GLM5 BF16 Lt plan M=%u K=%u N=4096 solution=1176 workspace=0 name=%s\n",
                m,k,name.c_str());
        ready = true;
    } while (false);
    if (!ready) {
        glm5_bf16_lt_destroy_plan(p);
        fprintf(stderr,"ds4: ROCm GLM5 BF16 Lt required plan unavailable M=%u K=%u\n",m,k);
        return nullptr;
    }
    if (!g_glm5_bf16_lt_activations &&
        hipMalloc(&g_glm5_bf16_lt_activations,glm5_bf16_lt_scratch_bytes) != hipSuccess) {
        glm5_bf16_lt_destroy_plan(p);
        return nullptr;
    }
    g_glm5_bf16_lt_plans.push_back(p);
    return &g_glm5_bf16_lt_plans.back();
}
