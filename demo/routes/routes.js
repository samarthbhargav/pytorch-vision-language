module.exports = function (express) {
    var router = express.Router();
    var PythonShell = require('python-shell');
    var api_root = '/api';
    var path = require('path');

    router.get(api_root, function (req, res) {
        res.json({ 'success' : true, 'message' : 'Welcome', 'status' : 'Online' });
    });

    // host images
    router.get('/image/*', function (req, res) {
        var path_list = req.url.split('/');
        path_list.shift(); path_list.shift();
        res.sendFile(path.join(__dirname, '../../data/', path_list.join('/')));
    });

    // classes
    router.get(api_root + '/classes', function (req, res) {
        PythonShell.run('api.py', {
            mode: 'json',
            pythonPath: '/anaconda/bin/python',
            pythonOptions: [ '-u' ],
            scriptPath: '../api',
            args: [ 'classes' ]
        }, function (err, results) {
            if (err) throw err;
            res.json({ 'success' : true, 'message' : results[0], 'status' : 'Online' });
        });
    });

    // counterfactual
    router.get(api_root + '/counterfactual', function (req, res) {
        PythonShell.run('api.py', {
            mode: 'json',
            pythonPath: '/anaconda/bin/python',
            pythonOptions: [ '-u' ],
            scriptPath: '../api',
            args: [ 'counterfactual', req.query.class_true, req.query.class_false ]
        }, function (err, results) {
            if (err) throw err;
            res.json({ 'success' : true, 'message' : results[0], 'status' : 'Online' });
        });
    });

    return router;
};
