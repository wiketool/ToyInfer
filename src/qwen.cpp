#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <stdexcept>

#include "config.h"
#include "cuda_utils.h"
#include "options.h"
#include "qwen3.h"
#include "spdlog/spdlog.h"
#include "type.h"

namespace toyinfer {
const Qwen3::TensorMeta Qwen3::TENSORMETA[] = {
    {"model.embed_tokens.weight", 0, 777912320},
    {"model.layers.0.input_layernorm.weight", 777912320, 5120},
    {"model.layers.0.mlp.down_proj.weight", 777917440, 49807360},
    {"model.layers.0.mlp.gate_proj.weight", 827724800, 49807360},
    {"model.layers.0.mlp.up_proj.weight", 877532160, 49807360},
    {"model.layers.0.post_attention_layernorm.weight", 927339520, 5120},
    {"model.layers.0.self_attn.k_norm.weight", 927344640, 256},
    {"model.layers.0.self_attn.k_proj.weight", 927344896, 5242880},
    {"model.layers.0.self_attn.o_proj.weight", 932587776, 20971520},
    {"model.layers.0.self_attn.q_norm.weight", 953559296, 256},
    {"model.layers.0.self_attn.q_proj.weight", 953559552, 20971520},
    {"model.layers.0.self_attn.v_proj.weight", 974531072, 5242880},
    {"model.layers.1.input_layernorm.weight", 979773952, 5120},
    {"model.layers.1.mlp.down_proj.weight", 979779072, 49807360},
    {"model.layers.1.mlp.gate_proj.weight", 1029586432, 49807360},
    {"model.layers.1.mlp.up_proj.weight", 1079393792, 49807360},
    {"model.layers.1.post_attention_layernorm.weight", 1129201152, 5120},
    {"model.layers.1.self_attn.k_norm.weight", 1129206272, 256},
    {"model.layers.1.self_attn.k_proj.weight", 1129206528, 5242880},
    {"model.layers.1.self_attn.o_proj.weight", 1134449408, 20971520},
    {"model.layers.1.self_attn.q_norm.weight", 1155420928, 256},
    {"model.layers.1.self_attn.q_proj.weight", 1155421184, 20971520},
    {"model.layers.1.self_attn.v_proj.weight", 1176392704, 5242880},
    {"model.layers.10.input_layernorm.weight", 1181635584, 5120},
    {"model.layers.10.mlp.down_proj.weight", 1181640704, 49807360},
    {"model.layers.10.mlp.gate_proj.weight", 1231448064, 49807360},
    {"model.layers.10.mlp.up_proj.weight", 1281255424, 49807360},
    {"model.layers.10.post_attention_layernorm.weight", 1331062784, 5120},
    {"model.layers.10.self_attn.k_norm.weight", 1331067904, 256},
    {"model.layers.10.self_attn.k_proj.weight", 1331068160, 5242880},
    {"model.layers.10.self_attn.o_proj.weight", 1336311040, 20971520},
    {"model.layers.10.self_attn.q_norm.weight", 1357282560, 256},
    {"model.layers.10.self_attn.q_proj.weight", 1357282816, 20971520},
    {"model.layers.10.self_attn.v_proj.weight", 1378254336, 5242880},
    {"model.layers.11.input_layernorm.weight", 1383497216, 5120},
    {"model.layers.11.mlp.down_proj.weight", 1383502336, 49807360},
    {"model.layers.11.mlp.gate_proj.weight", 1433309696, 49807360},
    {"model.layers.11.mlp.up_proj.weight", 1483117056, 49807360},
    {"model.layers.11.post_attention_layernorm.weight", 1532924416, 5120},
    {"model.layers.11.self_attn.k_norm.weight", 1532929536, 256},
    {"model.layers.11.self_attn.k_proj.weight", 1532929792, 5242880},
    {"model.layers.11.self_attn.o_proj.weight", 1538172672, 20971520},
    {"model.layers.11.self_attn.q_norm.weight", 1559144192, 256},
    {"model.layers.11.self_attn.q_proj.weight", 1559144448, 20971520},
    {"model.layers.11.self_attn.v_proj.weight", 1580115968, 5242880},
    {"model.layers.12.input_layernorm.weight", 1585358848, 5120},
    {"model.layers.12.mlp.down_proj.weight", 1585363968, 49807360},
    {"model.layers.12.mlp.gate_proj.weight", 1635171328, 49807360},
    {"model.layers.12.mlp.up_proj.weight", 1684978688, 49807360},
    {"model.layers.12.post_attention_layernorm.weight", 1734786048, 5120},
    {"model.layers.12.self_attn.k_norm.weight", 1734791168, 256},
    {"model.layers.12.self_attn.k_proj.weight", 1734791424, 5242880},
    {"model.layers.12.self_attn.o_proj.weight", 1740034304, 20971520},
    {"model.layers.12.self_attn.q_norm.weight", 1761005824, 256},
    {"model.layers.12.self_attn.q_proj.weight", 1761006080, 20971520},
    {"model.layers.12.self_attn.v_proj.weight", 1781977600, 5242880},
    {"model.layers.13.input_layernorm.weight", 1787220480, 5120},
    {"model.layers.13.mlp.down_proj.weight", 1787225600, 49807360},
    {"model.layers.13.mlp.gate_proj.weight", 1837032960, 49807360},
    {"model.layers.13.mlp.up_proj.weight", 1886840320, 49807360},
    {"model.layers.13.post_attention_layernorm.weight", 1936647680, 5120},
    {"model.layers.13.self_attn.k_norm.weight", 1936652800, 256},
    {"model.layers.13.self_attn.k_proj.weight", 1936653056, 5242880},
    {"model.layers.13.self_attn.o_proj.weight", 1941895936, 20971520},
    {"model.layers.13.self_attn.q_norm.weight", 1962867456, 256},
    {"model.layers.13.self_attn.q_proj.weight", 1962867712, 20971520},
    {"model.layers.13.self_attn.v_proj.weight", 1983839232, 5242880},
    {"model.layers.14.input_layernorm.weight", 1989082112, 5120},
    {"model.layers.14.mlp.down_proj.weight", 1989087232, 49807360},
    {"model.layers.14.mlp.gate_proj.weight", 2038894592, 49807360},
    {"model.layers.14.mlp.up_proj.weight", 2088701952, 49807360},
    {"model.layers.14.post_attention_layernorm.weight", 2138509312, 5120},
    {"model.layers.14.self_attn.k_norm.weight", 2138514432, 256},
    {"model.layers.14.self_attn.k_proj.weight", 2138514688, 5242880},
    {"model.layers.14.self_attn.o_proj.weight", 2143757568, 20971520},
    {"model.layers.14.self_attn.q_norm.weight", 2164729088, 256},
    {"model.layers.14.self_attn.q_proj.weight", 2164729344, 20971520},
    {"model.layers.14.self_attn.v_proj.weight", 2185700864, 5242880},
    {"model.layers.15.input_layernorm.weight", 3957880832, 5120},
    {"model.layers.15.mlp.down_proj.weight", 3957885952, 49807360},
    {"model.layers.15.mlp.gate_proj.weight", 2190943744, 49807360},
    {"model.layers.15.mlp.up_proj.weight", 2240751104, 49807360},
    {"model.layers.15.post_attention_layernorm.weight", 4007693312, 5120},
    {"model.layers.15.self_attn.k_norm.weight", 2290558464, 256},
    {"model.layers.15.self_attn.k_proj.weight", 2290558720, 5242880},
    {"model.layers.15.self_attn.o_proj.weight", 2295801600, 20971520},
    {"model.layers.15.self_attn.q_norm.weight", 2316773120, 256},
    {"model.layers.15.self_attn.q_proj.weight", 2316773376, 20971520},
    {"model.layers.15.self_attn.v_proj.weight", 2337744896, 5242880},
    {"model.layers.16.input_layernorm.weight", 4007698432, 5120},
    {"model.layers.16.mlp.down_proj.weight", 4007703552, 49807360},
    {"model.layers.16.mlp.gate_proj.weight", 4057510912, 49807360},
    {"model.layers.16.mlp.up_proj.weight", 4107318272, 49807360},
    {"model.layers.16.post_attention_layernorm.weight", 4157125632, 5120},
    {"model.layers.16.self_attn.k_norm.weight", 4157130752, 256},
    {"model.layers.16.self_attn.k_proj.weight", 4157131008, 5242880},
    {"model.layers.16.self_attn.o_proj.weight", 4162373888, 20971520},
    {"model.layers.16.self_attn.q_norm.weight", 4183345408, 256},
    {"model.layers.16.self_attn.q_proj.weight", 4183345664, 20971520},
    {"model.layers.16.self_attn.v_proj.weight", 4204317184, 5242880},
    {"model.layers.17.input_layernorm.weight", 4209560064, 5120},
    {"model.layers.17.mlp.down_proj.weight", 4209565184, 49807360},
    {"model.layers.17.mlp.gate_proj.weight", 4259372544, 49807360},
    {"model.layers.17.mlp.up_proj.weight", 4309179904, 49807360},
    {"model.layers.17.post_attention_layernorm.weight", 4358987264, 5120},
    {"model.layers.17.self_attn.k_norm.weight", 4358992384, 256},
    {"model.layers.17.self_attn.k_proj.weight", 4358992640, 5242880},
    {"model.layers.17.self_attn.o_proj.weight", 4364235520, 20971520},
    {"model.layers.17.self_attn.q_norm.weight", 4385207040, 256},
    {"model.layers.17.self_attn.q_proj.weight", 4385207296, 20971520},
    {"model.layers.17.self_attn.v_proj.weight", 4406178816, 5242880},
    {"model.layers.18.input_layernorm.weight", 4411421696, 5120},
    {"model.layers.18.mlp.down_proj.weight", 4411426816, 49807360},
    {"model.layers.18.mlp.gate_proj.weight", 4461234176, 49807360},
    {"model.layers.18.mlp.up_proj.weight", 4511041536, 49807360},
    {"model.layers.18.post_attention_layernorm.weight", 4560848896, 5120},
    {"model.layers.18.self_attn.k_norm.weight", 4560854016, 256},
    {"model.layers.18.self_attn.k_proj.weight", 4560854272, 5242880},
    {"model.layers.18.self_attn.o_proj.weight", 4566097152, 20971520},
    {"model.layers.18.self_attn.q_norm.weight", 4587068672, 256},
    {"model.layers.18.self_attn.q_proj.weight", 4587068928, 20971520},
    {"model.layers.18.self_attn.v_proj.weight", 4608040448, 5242880},
    {"model.layers.19.input_layernorm.weight", 4613283328, 5120},
    {"model.layers.19.mlp.down_proj.weight", 4613288448, 49807360},
    {"model.layers.19.mlp.gate_proj.weight", 4663095808, 49807360},
    {"model.layers.19.mlp.up_proj.weight", 4712903168, 49807360},
    {"model.layers.19.post_attention_layernorm.weight", 4762710528, 5120},
    {"model.layers.19.self_attn.k_norm.weight", 4762715648, 256},
    {"model.layers.19.self_attn.k_proj.weight", 4762715904, 5242880},
    {"model.layers.19.self_attn.o_proj.weight", 4767958784, 20971520},
    {"model.layers.19.self_attn.q_norm.weight", 4788930304, 256},
    {"model.layers.19.self_attn.q_proj.weight", 4788930560, 20971520},
    {"model.layers.19.self_attn.v_proj.weight", 4809902080, 5242880},
    {"model.layers.2.input_layernorm.weight", 2342987776, 5120},
    {"model.layers.2.mlp.down_proj.weight", 2342992896, 49807360},
    {"model.layers.2.mlp.gate_proj.weight", 2392800256, 49807360},
    {"model.layers.2.mlp.up_proj.weight", 2442607616, 49807360},
    {"model.layers.2.post_attention_layernorm.weight", 2492414976, 5120},
    {"model.layers.2.self_attn.k_norm.weight", 2492420096, 256},
    {"model.layers.2.self_attn.k_proj.weight", 2492420352, 5242880},
    {"model.layers.2.self_attn.o_proj.weight", 2497663232, 20971520},
    {"model.layers.2.self_attn.q_norm.weight", 2518634752, 256},
    {"model.layers.2.self_attn.q_proj.weight", 2518635008, 20971520},
    {"model.layers.2.self_attn.v_proj.weight", 2539606528, 5242880},
    {"model.layers.20.input_layernorm.weight", 4815144960, 5120},
    {"model.layers.20.mlp.down_proj.weight", 4815150080, 49807360},
    {"model.layers.20.mlp.gate_proj.weight", 4864957440, 49807360},
    {"model.layers.20.mlp.up_proj.weight", 4914764800, 49807360},
    {"model.layers.20.post_attention_layernorm.weight", 4964572160, 5120},
    {"model.layers.20.self_attn.k_norm.weight", 4964577280, 256},
    {"model.layers.20.self_attn.k_proj.weight", 4964577536, 5242880},
    {"model.layers.20.self_attn.o_proj.weight", 4969820416, 20971520},
    {"model.layers.20.self_attn.q_norm.weight", 4990791936, 256},
    {"model.layers.20.self_attn.q_proj.weight", 4990792192, 20971520},
    {"model.layers.20.self_attn.v_proj.weight", 5011763712, 5242880},
    {"model.layers.21.input_layernorm.weight", 5017006592, 5120},
    {"model.layers.21.mlp.down_proj.weight", 5017011712, 49807360},
    {"model.layers.21.mlp.gate_proj.weight", 5066819072, 49807360},
    {"model.layers.21.mlp.up_proj.weight", 5116626432, 49807360},
    {"model.layers.21.post_attention_layernorm.weight", 5166433792, 5120},
    {"model.layers.21.self_attn.k_norm.weight", 5166438912, 256},
    {"model.layers.21.self_attn.k_proj.weight", 5166439168, 5242880},
    {"model.layers.21.self_attn.o_proj.weight", 5171682048, 20971520},
    {"model.layers.21.self_attn.q_norm.weight", 5192653568, 256},
    {"model.layers.21.self_attn.q_proj.weight", 5192653824, 20971520},
    {"model.layers.21.self_attn.v_proj.weight", 5213625344, 5242880},
    {"model.layers.22.input_layernorm.weight", 5218868224, 5120},
    {"model.layers.22.mlp.down_proj.weight", 5218873344, 49807360},
    {"model.layers.22.mlp.gate_proj.weight", 5268680704, 49807360},
    {"model.layers.22.mlp.up_proj.weight", 5318488064, 49807360},
    {"model.layers.22.post_attention_layernorm.weight", 5368295424, 5120},
    {"model.layers.22.self_attn.k_norm.weight", 5368300544, 256},
    {"model.layers.22.self_attn.k_proj.weight", 5368300800, 5242880},
    {"model.layers.22.self_attn.o_proj.weight", 5373543680, 20971520},
    {"model.layers.22.self_attn.q_norm.weight", 5394515200, 256},
    {"model.layers.22.self_attn.q_proj.weight", 5394515456, 20971520},
    {"model.layers.22.self_attn.v_proj.weight", 5415486976, 5242880},
    {"model.layers.23.input_layernorm.weight", 5420729856, 5120},
    {"model.layers.23.mlp.down_proj.weight", 5420734976, 49807360},
    {"model.layers.23.mlp.gate_proj.weight", 5470542336, 49807360},
    {"model.layers.23.mlp.up_proj.weight", 5520349696, 49807360},
    {"model.layers.23.post_attention_layernorm.weight", 5570157056, 5120},
    {"model.layers.23.self_attn.k_norm.weight", 5570162176, 256},
    {"model.layers.23.self_attn.k_proj.weight", 5570162432, 5242880},
    {"model.layers.23.self_attn.o_proj.weight", 5575405312, 20971520},
    {"model.layers.23.self_attn.q_norm.weight", 5596376832, 256},
    {"model.layers.23.self_attn.q_proj.weight", 5596377088, 20971520},
    {"model.layers.23.self_attn.v_proj.weight", 5617348608, 5242880},
    {"model.layers.24.input_layernorm.weight", 5622591488, 5120},
    {"model.layers.24.mlp.down_proj.weight", 5622596608, 49807360},
    {"model.layers.24.mlp.gate_proj.weight", 5672403968, 49807360},
    {"model.layers.24.mlp.up_proj.weight", 5722211328, 49807360},
    {"model.layers.24.post_attention_layernorm.weight", 5772018688, 5120},
    {"model.layers.24.self_attn.k_norm.weight", 5772023808, 256},
    {"model.layers.24.self_attn.k_proj.weight", 5772024064, 5242880},
    {"model.layers.24.self_attn.o_proj.weight", 5777266944, 20971520},
    {"model.layers.24.self_attn.q_norm.weight", 5798238464, 256},
    {"model.layers.24.self_attn.q_proj.weight", 5798238720, 20971520},
    {"model.layers.24.self_attn.v_proj.weight", 5819210240, 5242880},
    {"model.layers.25.input_layernorm.weight", 5824453120, 5120},
    {"model.layers.25.mlp.down_proj.weight", 5824458240, 49807360},
    {"model.layers.25.mlp.gate_proj.weight", 5874265600, 49807360},
    {"model.layers.25.mlp.up_proj.weight", 5924072960, 49807360},
    {"model.layers.25.post_attention_layernorm.weight", 5973880320, 5120},
    {"model.layers.25.self_attn.k_norm.weight", 5973885440, 256},
    {"model.layers.25.self_attn.k_proj.weight", 5973885696, 5242880},
    {"model.layers.25.self_attn.o_proj.weight", 5979128576, 20971520},
    {"model.layers.25.self_attn.q_norm.weight", 6000100096, 256},
    {"model.layers.25.self_attn.q_proj.weight", 6000100352, 20971520},
    {"model.layers.25.self_attn.v_proj.weight", 6021071872, 5242880},
    {"model.layers.26.input_layernorm.weight", 6026314752, 5120},
    {"model.layers.26.mlp.down_proj.weight", 6026319872, 49807360},
    {"model.layers.26.mlp.gate_proj.weight", 6076127232, 49807360},
    {"model.layers.26.mlp.up_proj.weight", 6125934592, 49807360},
    {"model.layers.26.post_attention_layernorm.weight", 6175741952, 5120},
    {"model.layers.26.self_attn.k_norm.weight", 6175747072, 256},
    {"model.layers.26.self_attn.k_proj.weight", 6175747328, 5242880},
    {"model.layers.26.self_attn.o_proj.weight", 6180990208, 20971520},
    {"model.layers.26.self_attn.q_norm.weight", 6201961728, 256},
    {"model.layers.26.self_attn.q_proj.weight", 6201961984, 20971520},
    {"model.layers.26.self_attn.v_proj.weight", 6222933504, 5242880},
    {"model.layers.27.input_layernorm.weight", 6228176384, 5120},
    {"model.layers.27.mlp.down_proj.weight", 6228181504, 49807360},
    {"model.layers.27.mlp.gate_proj.weight", 6277988864, 49807360},
    {"model.layers.27.mlp.up_proj.weight", 6327796224, 49807360},
    {"model.layers.27.post_attention_layernorm.weight", 6377603584, 5120},
    {"model.layers.27.self_attn.k_norm.weight", 6377608704, 256},
    {"model.layers.27.self_attn.k_proj.weight", 6377608960, 5242880},
    {"model.layers.27.self_attn.o_proj.weight", 6382851840, 20971520},
    {"model.layers.27.self_attn.q_norm.weight", 6403823360, 256},
    {"model.layers.27.self_attn.q_proj.weight", 6403823616, 20971520},
    {"model.layers.27.self_attn.v_proj.weight", 6424795136, 5242880},
    {"model.layers.28.input_layernorm.weight", 6430038016, 5120},
    {"model.layers.28.mlp.down_proj.weight", 6430043136, 49807360},
    {"model.layers.28.mlp.gate_proj.weight", 6479850496, 49807360},
    {"model.layers.28.mlp.up_proj.weight", 6529657856, 49807360},
    {"model.layers.28.post_attention_layernorm.weight", 6579465216, 5120},
    {"model.layers.28.self_attn.k_norm.weight", 6579470336, 256},
    {"model.layers.28.self_attn.k_proj.weight", 6579470592, 5242880},
    {"model.layers.28.self_attn.o_proj.weight", 6584713472, 20971520},
    {"model.layers.28.self_attn.q_norm.weight", 6605684992, 256},
    {"model.layers.28.self_attn.q_proj.weight", 6605685248, 20971520},
    {"model.layers.28.self_attn.v_proj.weight", 6626656768, 5242880},
    {"model.layers.29.input_layernorm.weight", 6631899648, 5120},
    {"model.layers.29.mlp.down_proj.weight", 6631904768, 49807360},
    {"model.layers.29.mlp.gate_proj.weight", 6681712128, 49807360},
    {"model.layers.29.mlp.up_proj.weight", 6731519488, 49807360},
    {"model.layers.29.post_attention_layernorm.weight", 6781326848, 5120},
    {"model.layers.29.self_attn.k_norm.weight", 6781331968, 256},
    {"model.layers.29.self_attn.k_proj.weight", 6781332224, 5242880},
    {"model.layers.29.self_attn.o_proj.weight", 6786575104, 20971520},
    {"model.layers.29.self_attn.q_norm.weight", 6807546624, 256},
    {"model.layers.29.self_attn.q_proj.weight", 6807546880, 20971520},
    {"model.layers.29.self_attn.v_proj.weight", 6828518400, 5242880},
    {"model.layers.3.input_layernorm.weight", 2544849408, 5120},
    {"model.layers.3.mlp.down_proj.weight", 2544854528, 49807360},
    {"model.layers.3.mlp.gate_proj.weight", 2594661888, 49807360},
    {"model.layers.3.mlp.up_proj.weight", 2644469248, 49807360},
    {"model.layers.3.post_attention_layernorm.weight", 2694276608, 5120},
    {"model.layers.3.self_attn.k_norm.weight", 2694281728, 256},
    {"model.layers.3.self_attn.k_proj.weight", 2694281984, 5242880},
    {"model.layers.3.self_attn.o_proj.weight", 2699524864, 20971520},
    {"model.layers.3.self_attn.q_norm.weight", 2720496384, 256},
    {"model.layers.3.self_attn.q_proj.weight", 2720496640, 20971520},
    {"model.layers.3.self_attn.v_proj.weight", 2741468160, 5242880},
    {"model.layers.30.input_layernorm.weight", 6833761280, 5120},
    {"model.layers.30.mlp.down_proj.weight", 6833766400, 49807360},
    {"model.layers.30.mlp.gate_proj.weight", 6883573760, 49807360},
    {"model.layers.30.mlp.up_proj.weight", 6933381120, 49807360},
    {"model.layers.30.post_attention_layernorm.weight", 6983188480, 5120},
    {"model.layers.30.self_attn.k_norm.weight", 6983193600, 256},
    {"model.layers.30.self_attn.k_proj.weight", 6983193856, 5242880},
    {"model.layers.30.self_attn.o_proj.weight", 6988436736, 20971520},
    {"model.layers.30.self_attn.q_norm.weight", 7009408256, 256},
    {"model.layers.30.self_attn.q_proj.weight", 7009408512, 20971520},
    {"model.layers.30.self_attn.v_proj.weight", 7030380032, 5242880},
    {"model.layers.31.input_layernorm.weight", 7035622912, 5120},
    {"model.layers.31.mlp.down_proj.weight", 7035628032, 49807360},
    {"model.layers.31.mlp.gate_proj.weight", 7085435392, 49807360},
    {"model.layers.31.mlp.up_proj.weight", 7135242752, 49807360},
    {"model.layers.31.post_attention_layernorm.weight", 7185050112, 5120},
    {"model.layers.31.self_attn.k_norm.weight", 7185055232, 256},
    {"model.layers.31.self_attn.k_proj.weight", 7185055488, 5242880},
    {"model.layers.31.self_attn.o_proj.weight", 7190298368, 20971520},
    {"model.layers.31.self_attn.q_norm.weight", 7211269888, 256},
    {"model.layers.31.self_attn.q_proj.weight", 7211270144, 20971520},
    {"model.layers.31.self_attn.v_proj.weight", 7232241664, 5242880},
    {"model.layers.32.input_layernorm.weight", 7237484544, 5120},
    {"model.layers.32.mlp.down_proj.weight", 7237489664, 49807360},
    {"model.layers.32.mlp.gate_proj.weight", 7287297024, 49807360},
    {"model.layers.32.mlp.up_proj.weight", 7337104384, 49807360},
    {"model.layers.32.post_attention_layernorm.weight", 7386911744, 5120},
    {"model.layers.32.self_attn.k_norm.weight", 7386916864, 256},
    {"model.layers.32.self_attn.k_proj.weight", 7386917120, 5242880},
    {"model.layers.32.self_attn.o_proj.weight", 7392160000, 20971520},
    {"model.layers.32.self_attn.q_norm.weight", 7413131520, 256},
    {"model.layers.32.self_attn.q_proj.weight", 7413131776, 20971520},
    {"model.layers.32.self_attn.v_proj.weight", 7434103296, 5242880},
    {"model.layers.33.input_layernorm.weight", 7439346176, 5120},
    {"model.layers.33.mlp.down_proj.weight", 7439351296, 49807360},
    {"model.layers.33.mlp.gate_proj.weight", 7489158656, 49807360},
    {"model.layers.33.mlp.up_proj.weight", 7538966016, 49807360},
    {"model.layers.33.post_attention_layernorm.weight", 7588773376, 5120},
    {"model.layers.33.self_attn.k_norm.weight", 7588778496, 256},
    {"model.layers.33.self_attn.k_proj.weight", 7588778752, 5242880},
    {"model.layers.33.self_attn.o_proj.weight", 7594021632, 20971520},
    {"model.layers.33.self_attn.q_norm.weight", 7614993152, 256},
    {"model.layers.33.self_attn.q_proj.weight", 7614993408, 20971520},
    {"model.layers.33.self_attn.v_proj.weight", 7635964928, 5242880},
    {"model.layers.34.input_layernorm.weight", 7641207808, 5120},
    {"model.layers.34.mlp.down_proj.weight", 7641212928, 49807360},
    {"model.layers.34.mlp.gate_proj.weight", 7691020288, 49807360},
    {"model.layers.34.mlp.up_proj.weight", 7740827648, 49807360},
    {"model.layers.34.post_attention_layernorm.weight", 7790635008, 5120},
    {"model.layers.34.self_attn.k_norm.weight", 7790640128, 256},
    {"model.layers.34.self_attn.k_proj.weight", 7790640384, 5242880},
    {"model.layers.34.self_attn.o_proj.weight", 7795883264, 20971520},
    {"model.layers.34.self_attn.q_norm.weight", 7816854784, 256},
    {"model.layers.34.self_attn.q_proj.weight", 7816855040, 20971520},
    {"model.layers.34.self_attn.v_proj.weight", 7837826560, 5242880},
    {"model.layers.35.input_layernorm.weight", 7945306112, 5120},
    {"model.layers.35.mlp.down_proj.weight", 7945311232, 49807360},
    {"model.layers.35.mlp.gate_proj.weight", 7843069440, 49807360},
    {"model.layers.35.mlp.up_proj.weight", 7995118592, 49807360},
    {"model.layers.35.post_attention_layernorm.weight", 8044925952, 5120},
    {"model.layers.35.self_attn.k_norm.weight", 7892876800, 256},
    {"model.layers.35.self_attn.k_proj.weight", 7892877056, 5242880},
    {"model.layers.35.self_attn.o_proj.weight", 7898119936, 20971520},
    {"model.layers.35.self_attn.q_norm.weight", 7919091456, 256},
    {"model.layers.35.self_attn.q_proj.weight", 7919091712, 20971520},
    {"model.layers.35.self_attn.v_proj.weight", 7940063232, 5242880},
    {"model.layers.4.input_layernorm.weight", 2746711040, 5120},
    {"model.layers.4.mlp.down_proj.weight", 2746716160, 49807360},
    {"model.layers.4.mlp.gate_proj.weight", 2796523520, 49807360},
    {"model.layers.4.mlp.up_proj.weight", 2846330880, 49807360},
    {"model.layers.4.post_attention_layernorm.weight", 2896138240, 5120},
    {"model.layers.4.self_attn.k_norm.weight", 2896143360, 256},
    {"model.layers.4.self_attn.k_proj.weight", 2896143616, 5242880},
    {"model.layers.4.self_attn.o_proj.weight", 2901386496, 20971520},
    {"model.layers.4.self_attn.q_norm.weight", 2922358016, 256},
    {"model.layers.4.self_attn.q_proj.weight", 2922358272, 20971520},
    {"model.layers.4.self_attn.v_proj.weight", 2943329792, 5242880},
    {"model.layers.5.input_layernorm.weight", 2948572672, 5120},
    {"model.layers.5.mlp.down_proj.weight", 2948577792, 49807360},
    {"model.layers.5.mlp.gate_proj.weight", 2998385152, 49807360},
    {"model.layers.5.mlp.up_proj.weight", 3048192512, 49807360},
    {"model.layers.5.post_attention_layernorm.weight", 3097999872, 5120},
    {"model.layers.5.self_attn.k_norm.weight", 3098004992, 256},
    {"model.layers.5.self_attn.k_proj.weight", 3098005248, 5242880},
    {"model.layers.5.self_attn.o_proj.weight", 3103248128, 20971520},
    {"model.layers.5.self_attn.q_norm.weight", 3124219648, 256},
    {"model.layers.5.self_attn.q_proj.weight", 3124219904, 20971520},
    {"model.layers.5.self_attn.v_proj.weight", 3145191424, 5242880},
    {"model.layers.6.input_layernorm.weight", 3150434304, 5120},
    {"model.layers.6.mlp.down_proj.weight", 3150439424, 49807360},
    {"model.layers.6.mlp.gate_proj.weight", 3200246784, 49807360},
    {"model.layers.6.mlp.up_proj.weight", 3250054144, 49807360},
    {"model.layers.6.post_attention_layernorm.weight", 3299861504, 5120},
    {"model.layers.6.self_attn.k_norm.weight", 3299866624, 256},
    {"model.layers.6.self_attn.k_proj.weight", 3299866880, 5242880},
    {"model.layers.6.self_attn.o_proj.weight", 3305109760, 20971520},
    {"model.layers.6.self_attn.q_norm.weight", 3326081280, 256},
    {"model.layers.6.self_attn.q_proj.weight", 3326081536, 20971520},
    {"model.layers.6.self_attn.v_proj.weight", 3347053056, 5242880},
    {"model.layers.7.input_layernorm.weight", 3352295936, 5120},
    {"model.layers.7.mlp.down_proj.weight", 3352301056, 49807360},
    {"model.layers.7.mlp.gate_proj.weight", 3402108416, 49807360},
    {"model.layers.7.mlp.up_proj.weight", 3451915776, 49807360},
    {"model.layers.7.post_attention_layernorm.weight", 3501723136, 5120},
    {"model.layers.7.self_attn.k_norm.weight", 3501728256, 256},
    {"model.layers.7.self_attn.k_proj.weight", 3501728512, 5242880},
    {"model.layers.7.self_attn.o_proj.weight", 3506971392, 20971520},
    {"model.layers.7.self_attn.q_norm.weight", 3527942912, 256},
    {"model.layers.7.self_attn.q_proj.weight", 3527943168, 20971520},
    {"model.layers.7.self_attn.v_proj.weight", 3548914688, 5242880},
    {"model.layers.8.input_layernorm.weight", 3554157568, 5120},
    {"model.layers.8.mlp.down_proj.weight", 3554162688, 49807360},
    {"model.layers.8.mlp.gate_proj.weight", 3603970048, 49807360},
    {"model.layers.8.mlp.up_proj.weight", 3653777408, 49807360},
    {"model.layers.8.post_attention_layernorm.weight", 3703584768, 5120},
    {"model.layers.8.self_attn.k_norm.weight", 3703589888, 256},
    {"model.layers.8.self_attn.k_proj.weight", 3703590144, 5242880},
    {"model.layers.8.self_attn.o_proj.weight", 3708833024, 20971520},
    {"model.layers.8.self_attn.q_norm.weight", 3729804544, 256},
    {"model.layers.8.self_attn.q_proj.weight", 3729804800, 20971520},
    {"model.layers.8.self_attn.v_proj.weight", 3750776320, 5242880},
    {"model.layers.9.input_layernorm.weight", 3756019200, 5120},
    {"model.layers.9.mlp.down_proj.weight", 3756024320, 49807360},
    {"model.layers.9.mlp.gate_proj.weight", 3805831680, 49807360},
    {"model.layers.9.mlp.up_proj.weight", 3855639040, 49807360},
    {"model.layers.9.post_attention_layernorm.weight", 3905446400, 5120},
    {"model.layers.9.self_attn.k_norm.weight", 3905451520, 256},
    {"model.layers.9.self_attn.k_proj.weight", 3905451776, 5242880},
    {"model.layers.9.self_attn.o_proj.weight", 3910694656, 20971520},
    {"model.layers.9.self_attn.q_norm.weight", 3931666176, 256},
    {"model.layers.9.self_attn.q_proj.weight", 3931666432, 20971520},
    {"model.layers.9.self_attn.v_proj.weight", 3952637952, 5242880},
    {"model.norm.weight", 8044931072, 5120},
};
Qwen3::Qwen3(const Options& options, const LLMConfig& config)
    : model_dir(options.model_dir), layers(config.num_hidden_layers) {
    layer = new Layer[layers];
}

Qwen3::~Qwen3() {
    if (layer) {
        delete[] layer;
    }
    if (weight_d) {
        cudaFree(weight_d);
    }
}

void Qwen3::load_weights() {
    constexpr uint32_t TOTAL_TENSOR_NUMBER = 3;
    constexpr uint32_t SAFETENSOR_FILENAME_LEN = 64;
    const char* safetensor_file_template = "model-%05d-of-%05d.safetensors";
    char safetensor_filename[SAFETENSOR_FILENAME_LEN];
    uint64_t weight_total_byte = 0;
    for (uint32_t i = 0; i < TOTAL_TENSOR_NUMBER; i++) {
        snprintf(safetensor_filename, SAFETENSOR_FILENAME_LEN,
                 safetensor_file_template, i + 1, TOTAL_TENSOR_NUMBER);
        std::filesystem::path safetensor_path = model_dir / safetensor_filename;
        std::ifstream safetensor_ifs{safetensor_path};
        if (safetensor_ifs.is_open() == false) {
            throw std::runtime_error("Unable to open safetensor file:" +
                                     safetensor_path.string());
        }
        uint64_t meta_len;
        safetensor_ifs.read((char*)&meta_len, sizeof(meta_len));
        safetensor_ifs.seekg(0, std::ios::end);
        uint64_t weight_bytes =
            (uint64_t)safetensor_ifs.tellg() - sizeof(meta_len) - meta_len;
        weight_total_byte += weight_bytes;
    }
    SPDLOG_INFO("Qwen3 Weight total bytes: {}", weight_total_byte);
    char* weight_h;
    CHECK_CUDA(cudaMallocHost(&weight_h, weight_total_byte));
    uint64_t cur_offset = 0;
    for (uint32_t i = 0; i < TOTAL_TENSOR_NUMBER; i++) {
        snprintf(safetensor_filename, SAFETENSOR_FILENAME_LEN,
                 safetensor_file_template, i + 1, TOTAL_TENSOR_NUMBER);
        std::filesystem::path safetensor_path = model_dir / safetensor_filename;
        std::ifstream safetensor_ifs{safetensor_path};
        if (safetensor_ifs.is_open() == false) {
            throw std::runtime_error("Unable to open safetensor file: {}" +
                                     safetensor_path.string());
        }
        uint64_t meta_len;
        safetensor_ifs.read((char*)&meta_len, sizeof(meta_len));
        safetensor_ifs.seekg(sizeof(meta_len) + meta_len, std::ios::beg);
        uint64_t weight_bytes =
            (uint64_t)safetensor_ifs.tellg() - sizeof(meta_len) - meta_len;
        safetensor_ifs.read(weight_h + cur_offset, weight_bytes);
        SPDLOG_INFO("DONE: read qwen3 weight from {}.",
                    safetensor_path.string());
    }
    CHECK_CUDA(cudaMalloc(&weight_d, weight_total_byte));
    CHECK_CUDA(cudaMemcpy(weight_d, weight_h, weight_total_byte,
                          cudaMemcpyHostToDevice));
    SPDLOG_INFO("DONE: Copy Qwen3 weight from host to GPU.");

    lmhead_d = (bf16*)(weight_d + TENSORMETA[0].offset);
    embed_tokens_d = (bf16*)(weight_d + TENSORMETA[0].offset);
    norm_d = (bf16*)(weight_d + TENSORMETA[397].offset);

    const int layer_map[] = {0,  1,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             2,  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
                             30, 31, 32, 33, 34, 35, 4,  5,  6,  7,  8,  9};
    const int tensors_per_layer = 11;
    for (uint32_t i = 0; i < sizeof(layer_map) / sizeof(int); i++) {
        uint32_t layer_idx = layer_map[i];
        uint32_t meta_idx = 1 + i * tensors_per_layer;
        layer[i].input_layernorm_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 0].offset);
        layer[i].ffn.down_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 1].offset);
        layer[i].ffn.gate_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 2].offset);
        layer[i].ffn.up_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 3].offset);
        layer[i].post_attention_layernorm_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 4].offset);
        layer[i].attention.k_norm_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 5].offset);
        layer[i].attention.k_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 6].offset);
        layer[i].attention.o_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 7].offset);
        layer[i].attention.q_norm_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 8].offset);
        layer[i].attention.q_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 9].offset);
        layer[i].attention.v_proj_d =
            (bf16*)(weight_d + TENSORMETA[meta_idx + 10].offset);
    }

    CHECK_CUDA(cudaFreeHost(weight_h));
};
}  // namespace toyinfer