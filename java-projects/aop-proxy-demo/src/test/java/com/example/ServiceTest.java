package com.example;

import com.example.service.UserService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.junit.jupiter.SpringJUnitConfig;

import static org.junit.jupiter.api.Assertions.*;

@SpringJUnitConfig(ApplicationConfig.class)
public class ServiceTest {

    @Autowired
    private UserService userService;

    @Test
    public void testProxyDetection() {
        // Verify that the UserService is wrapped in a proxy
        String className = userService.getClass().getName();
        System.out.println("UserService class: " + className);
        
        assertTrue(className.contains("Proxy"), 
            "UserService should be wrapped in a proxy");
        
        // Print proxy information
        System.out.println("\n=== Proxy Information ===");
        System.out.println("UserService class: " + className);
        System.out.println("Is proxy: " + className.contains("Proxy"));
        System.out.println("Proxy type: JDK Dynamic Proxy");
    }

    @Test
    public void testServiceMethods() {
        // Test successful operations
        userService.createUser("TestUser");
        String user = userService.getUser(1L);
        assertEquals("TestUser", user);
        
        // Test exception handling and capture stack trace
        try {
            userService.getUser(-1L);
            fail("Expected IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // Print full stack trace to demonstrate proxy pattern
            System.out.println("\n=== Stack Trace with Proxy ===");
            e.printStackTrace();
            System.out.println("=== End Stack Trace ===");
        }
    }

    @Test
    public void testStackFrameContainsProxy() {
        try {
            userService.getUser(-1L);
        } catch (Exception e) {
            // Check that the stack trace contains proxy-related frames
            StackTraceElement[] stackTrace = e.getStackTrace();
            boolean foundProxy = false;
            boolean foundAop = false;
            
            for (StackTraceElement element : stackTrace) {
                String className = element.getClassName();
                if (className.contains("Proxy")) {
                    foundProxy = true;
                }
                if (className.contains("aop") || className.contains("Aspect")) {
                    foundAop = true;
                }
            }
            
            assertTrue(foundProxy, "Stack trace should contain proxy class");
            assertTrue(foundAop, "Stack trace should contain AOP framework classes");
        }
    }
}