#include "logger.h"

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

void init_logger() {
    try {
        // 创建控制台 sink (多线程安全)
        // auto console_sink =
        //     std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        // console_sink->set_level(spdlog::level::debug);

        // 创建文件 sink，每个文件最大 5MB，最多保留 3 个文件
        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            "logs/app.log", 1048576 * 5, 3);
        file_sink->set_level(spdlog::level::info);

        // 将多个 sink 组合到一个 logger 中
        std::vector<spdlog::sink_ptr> sinks{file_sink};
        auto logger = std::make_shared<spdlog::logger>(
            "multi_sink", sinks.begin(), sinks.end());

        // 设置全局默认 logger，之后可以直接使用 spdlog::info() 等简写
        spdlog::set_default_logger(logger);

        // 当遇到 error 及以上级别时，立即刷新日志到磁盘
        spdlog::flush_on(spdlog::level::err);

    } catch (const spdlog::spdlog_ex& ex) {
        std::cout << "Log initialization failed: " << ex.what() << std::endl;
    }
}