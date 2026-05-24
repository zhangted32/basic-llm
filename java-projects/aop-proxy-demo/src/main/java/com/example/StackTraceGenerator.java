package com.example;

import com.example.service.UserService;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class StackTraceGenerator {

    public static void main(String[] args) {
        try (AnnotationConfigApplicationContext context = 
                new AnnotationConfigApplicationContext(ApplicationConfig.class)) {
            
            UserService userService = context.getBean(UserService.class);
            
            // Generate multiple stack traces
            for (int i = 0; i < 5; i++) {
                try {
                    // Different error scenarios
                    switch (i) {
                        case 0:
                            userService.getUser(null);
                            break;
                        case 1:
                            userService.getUser(-1L);
                            break;
                        case 2:
                            userService.createUser(null);
                            break;
                        case 3:
                            userService.getUser(999L); // User not found
                            break;
                        case 4:
                            userService.createUser(""); // Empty name
                            break;
                    }
                } catch (Exception e) {
                    saveStackTrace(e, i);
                    saveContext(userService);
                }
            }
            
            System.out.println("Stack traces generated successfully!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void saveStackTrace(Exception e, int index) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String fileName = String.format("trace_%s_%03d.json", timestamp, index);
        
        try (PrintWriter writer = new PrintWriter(new FileWriter("../../data/raw_traces/aop-proxy-demo/" + fileName))) {
            StringBuilder sb = new StringBuilder();
            sb.append("{\"timestamp\": \"").append(timestamp).append("\",\n");
            sb.append("\"trace\": \"");
            
            // Convert stack trace to escaped string
            StackTraceElement[] elements = e.getStackTrace();
            sb.append(escapeJson(e.toString()));
            for (StackTraceElement element : elements) {
                sb.append("\\n    at ").append(escapeJson(element.toString()));
            }
            
            // Include cause chain
            Throwable cause = e.getCause();
            while (cause != null) {
                sb.append("\\nCaused by: ").append(escapeJson(cause.toString()));
                StackTraceElement[] causeElements = cause.getStackTrace();
                for (StackTraceElement element : causeElements) {
                    sb.append("\\n    at ").append(escapeJson(element.toString()));
                }
                cause = cause.getCause();
            }
            
            sb.append("\",\n");
            sb.append("\"exception_type\": \"").append(e.getClass().getName()).append("\",\n");
            sb.append("\"message\": \"").append(escapeJson(e.getMessage())).append("\"\n");
            sb.append("}");
            
            writer.println(sb.toString());
            System.out.println("Saved: " + fileName);
            
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private static void saveContext(UserService userService) {
        try (PrintWriter writer = new PrintWriter(new FileWriter("../../data/raw_traces/aop-proxy-demo/context.json"))) {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  \"classes\": {\n");
            sb.append("    \"com.example.service.UserServiceImpl\": {\n");
            sb.append("      \"annotations\": [\"org.springframework.stereotype.Service\",\n");
            sb.append("                      \"org.springframework.transaction.annotation.Transactional\"],\n");
            sb.append("      \"methods\": {\n");
            sb.append("        \"getUser\": {\"annotations\": [\"@Transactional\"]},\n");
            sb.append("        \"createUser\": {\"annotations\": [\"@Transactional\"]}\n");
            sb.append("      }\n");
            sb.append("    },\n");
            sb.append("    \"com.example.repository.UserRepository\": {\n");
            sb.append("      \"annotations\": [\"org.springframework.stereotype.Repository\"],\n");
            sb.append("      \"methods\": {}\n");
            sb.append("    }\n");
            sb.append("  },\n");
            sb.append("  \"proxy_mappings\": {\n");
            sb.append("    \"$Proxy56\": \"com.example.service.UserServiceImpl\"\n");
            sb.append("  },\n");
            sb.append("  \"proxy_type\": \"").append(getProxyType(userService)).append("\"\n");
            sb.append("}\n");
            
            writer.println(sb.toString());
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private static String getProxyType(UserService userService) {
        String className = userService.getClass().getName();
        if (className.contains("$Proxy")) {
            return "jdk_proxy";
        } else if (className.contains("EnhancerByCGLIB")) {
            return "cglib";
        } else if (className.contains("$$Enhancer")) {
            return "spring_cglib";
        }
        return "unknown";
    }

    private static String escapeJson(String input) {
        if (input == null) {
            return "";
        }
        return input.replace("\\", "\\\\")
                   .replace("\"", "\\\"")
                   .replace("\n", "\\n")
                   .replace("\r", "\\r")
                   .replace("\t", "\\t");
    }
}