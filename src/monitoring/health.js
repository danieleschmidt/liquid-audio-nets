// Health Check and Monitoring Module
// Generated: 2025-08-31T03:17:29.703Z

const os = require('os');

class HealthMonitor {
    constructor() {
        this.startTime = Date.now();
        this.requestCount = 0;
        this.errorCount = 0;
        this.responseTime = [];
    }

    recordRequest(duration) {
        this.requestCount++;
        this.responseTime.push(duration);
        // Keep only last 1000 response times
        if (this.responseTime.length > 1000) {
            this.responseTime.shift();
        }
    }

    recordError() {
        this.errorCount++;
    }

    getHealth() {
        const uptime = Date.now() - this.startTime;
        const avgResponseTime = this.responseTime.length > 0 
            ? this.responseTime.reduce((a, b) => a + b, 0) / this.responseTime.length 
            : 0;

        return {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: Math.floor(uptime / 1000),
            metrics: {
                requestCount: this.requestCount,
                errorCount: this.errorCount,
                errorRate: this.requestCount > 0 ? (this.errorCount / this.requestCount * 100).toFixed(2) + '%' : '0%',
                avgResponseTime: Math.round(avgResponseTime) + 'ms',
                memory: {
                    used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024) + 'MB',
                    total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024) + 'MB'
                },
                cpu: {
                    usage: process.cpuUsage(),
                    loadAverage: os.loadavg()
                }
            }
        };
    }

    getReadiness() {
        // Check if service is ready to accept traffic
        const checks = {
            database: true, // TODO: Implement actual DB check
            cache: true,    // TODO: Implement actual cache check
            api: true       // TODO: Implement actual API check
        };

        const ready = Object.values(checks).every(check => check === true);

        return {
            ready,
            checks,
            timestamp: new Date().toISOString()
        };
    }
}

module.exports = new HealthMonitor();
