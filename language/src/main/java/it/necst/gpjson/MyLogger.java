package it.necst.gpjson;

import java.util.logging.*;

public class MyLogger {
    static Logger logger;
    StreamHandler handler;

    private MyLogger() {
        System.setProperty("java.util.logging.SimpleFormatter.format", "[%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS] %4$s: %5$s%n");
        logger = Logger.getLogger(MyLogger.class.getName());
        handler = new StreamHandler(System.out, new SimpleFormatter()) {
            @Override
            public synchronized void publish(final LogRecord record) {
                super.publish(record);
                flush();
            }
        };
        logger.setUseParentHandlers(false);
        logger.addHandler(handler);
    }

    private static Logger getLogger(){
        if (logger == null) {
            new MyLogger();
        }
        return logger;
    }

    public static void log(Level level, String sourceClass, String sourceMethod, String msg){
        getLogger().logp(level, sourceClass, sourceMethod, msg);
    }

    public static void setLevel(Level level) {
        getLogger().setLevel(level);
        for (Handler handler: getLogger().getHandlers()) {
            handler.setLevel(level);
        }
    }
}

