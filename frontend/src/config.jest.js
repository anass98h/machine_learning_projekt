module.exports = {
    transform: {
        "^.+\\.(js|jsx)$": "babel-jest",
    },
    moduleNameMapper: {
        "^axios$": "axios/dist/node/axios.cjs",  // Forza Jest a usare la versione CommonJS di axios
        "^react-gauge-chart$": "<rootDir>/__mocks__/react-gauge-chart.js" // Mock di react-gauge-chart
    },
    testEnvironment: "jsdom",
};
